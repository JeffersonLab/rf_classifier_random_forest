import os
import sys
import traceback
import math
import utils
import json
import joblib
from datetime import datetime

from base_model import BaseModel
from rfwtools.example import WindowedExample
from rfwtools.example_validator import WindowedExampleValidator
from rfwtools.extractor.autoregressive import autoregressive_extractor, get_signal_names
from rfwtools.config import Config

app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
"""The base directory of this model application."""

lib_dir = os.path.join(app_dir, 'lib')
"""The directory where python code and pickle files containing tsfresh models, etc. can be found."""


class Model(BaseModel):
    """
    This model uses random forest models to identify the faulted cavity and fault type of a C100 event.

    This model is based on work done by Chris Tennant, Lasitha Vidyaratne, Tom Powers, etc. and represents a improved
    model used to identify which cavity and fault type is associated with a C100 fault event.  Any individual cavity can
    be identified as the offending cavity.  Any collection of multiple cavities faulting at the same time are given the
    generic label of 'multiple'.  The following fault types may be identified by the model: Controls Fault, E_Quench,
    Heat Riser Choke, Microphonics, Quench_100ms, Quench_3ms, Single Cav Turn off, and Multi-cav Turn off.

    Additional documentation is available in the package docs folder.
    """

    def __init__(self, path):
        """Create a Model object.  This performs all data handling, validation, and analysis."""
        # Make sure we do not have a trailing slash to muck up processing later.
        path = path.rstrip(os.sep)
        super().__init__(path)

        # Split up the path into it's constituent pieces
        tokens = path.split(os.sep)
        dt = datetime.strptime(f"{tokens[-2]} {tokens[-1]}", "%Y_%m_%d %H%M%S.%f")
        zone = tokens[-3]

        # Save the root data path into the rfwtools configuration
        data_dir = os.sep + os.path.join(*tokens[:-3])
        Config().data_dir = data_dir

        self.example = WindowedExample(zone=zone, dt=dt, start=-1536.0, n_samples=7666, cavity_conf=math.nan,
                                       fault_conf=math.nan, cavity_label="", fault_label="", label_source="")
        self.example.load_data()
        self.validator = WindowedExampleValidator()
        self.common_features_df = None

    def analyze(self, deployment='ops'):
        """A method for analyzing the data held in event_dir that returns cavity and fault type label information.

        This method validates that the capture files and waveform data in event_dir are in the expected format and
        internally consistent.  First the cavity label is determined by the cavity model.  Should this return a
        "multiple" cavity label, the no fault type label determination is made.  Instead, a fault type label of
        "Multi Cav Turn Off" with the same confidence as the cavity label.
        """
        # Check that the data we're about to analyze meets any preconditions for our model
        self.validate_data(deployment)

        # Fault and cavity models use same data and features.  Get that now.
        signals = get_signal_names(cavities=['1', '2', '3', '4', '5', '6', '7', '8'],
                                   waveforms=['GMES', 'GASK', 'CRFP', 'DETA2'])
        self.common_features_df = autoregressive_extractor(self.example, normalize=True, max_lag=7, signals=signals)

        # Analyze the data to determine which cavity caused the fault.
        cav_results = self.get_cavity_label()

        # A value of cavity-label '0' corresponds to a multi-cavity event.  In this case the fault analysis is
        # unreliable and we should short circuit and report only a multi-cavity fault type (likely someone
        # performing a zone wide operation triggering a "fault").  Use the cavity confidence since it is the
        # prediction we're basing this on.
        fault_results = {'fault-label': 'Multi Cav turn off', 'fault-confidence': cav_results['cavity-confidence']}
        if cav_results['cavity-label'] != 'multiple':
            fault_results = self.get_fault_type_label(int(cav_results['cavity-label']))

        return {
            'location': self.example.event_zone,
            'timestamp': self.example.event_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-5],
            'cavity-label': cav_results['cavity-label'],
            'cavity-confidence': cav_results['cavity-confidence'],
            'fault-label': fault_results['fault-label'],
            'fault-confidence': fault_results['fault-confidence']
        }

    def validate_data(self, deployment='ops'):
        """Check that the event directory and it's data is of the expected format.

        This method inspects the event directory and raises an exception if a problem is found.  The following aspects
        of the event directory and waveform data are validated.
           # All eight cavities are represented by exactly one capture file
           # All of the required waveforms are represented exactly once
           # All of the capture files use the same timespan and have constant sampling intervals
           # All of the cavity are in the appropriate control mode (GDR I/Q => 4 or SELAP => 64)

        Args:
            deployment (str):  Which MYA deployment to use when validating cavity operating modes.

        Returns:
            None: Subroutines raise an exception if an error condition is found.
        """
        self.validator.set_example(self.example)

        # Don't just use the built in validate_data method as this needs to be future proofed against C100 firmware
        # upgrades.  This upgrade will result in a new mode SELAP (R...CNTL2MODE == 64).
        self.validator.validate_capture_file_counts()
        self.validator.validate_capture_file_waveforms()

        # Many of these examples will have some amount of rounding error.
        self.validator.validate_waveform_times(min_end=1532.9, max_start=0, step_size=0.2)
        self.validator.validate_cavity_modes(mode=(4, 64), deployment=deployment)
        self.validator.validate_zones()

    def get_cavity_label(self):
        """Loads the underlying cavity model and performs the predictions based on the common_features_df.

            Returns:
                A dictionary with format {'cavity-label': <string_label>, 'cavity-confidence': <float in [0,1]>}"
        """
        # Load the cavity model
        rf_cav_model = joblib.load(os.path.join(lib_dir, "model_files", 'RF_CAVITY_20210715.pkl'))

        # Load the model from disk and make a prediction about which cavity faulted first.  The predict() method returns
        # an array of results.  We only have one result, so pull it out of the array structure now
        cavity_id = rf_cav_model.predict(self.common_features_df)
        cavity_id = cavity_id[0]

        # predict_proba returns a mildly complicated 2D np.array structure.  First index is for each supplied example.
        # Second index is for the predicted probabilities for each category indexed on classes_.   We have one example,
        # and want to probability for the class corresponding to cavity_id (the actual prediction, i.e., the class with
        # the greatest probability).
        cavity_confidence = rf_cav_model.predict_proba(self.common_features_df)[0][cavity_id]

        # Convert the results from an int to a human readable string
        if cavity_id == 0:
            cavity_id = 'multiple'
        else:
            # The cavity_id int corresponds to the actual cavity number if 1-8
            cavity_id = str(cavity_id)

        return {'cavity-label': cavity_id, 'cavity-confidence': cavity_confidence}

    def get_fault_type_label(self, cavity_number):
        """Loads the underlying fault type model and performs the predictions based on the common_features_df.

            Args:
                cavity_number (int): The number of the cavity (1-8) that caused the fault.

            Returns:
                A dictionary with format {'fault-label': <string_label>, 'fault-confidence': <float in [0,1]>}"
        """
        # Make sure we received a valid cavity number
        self.assert_valid_cavity_number(cavity_number)

        # Imputing on a single example is useless since there is no population to provide ranges or median values
        # Load the fault type model and make a prediction about the type of fault
        rf_fault_model = joblib.load(os.path.join(lib_dir, 'model_files', 'RF_FAULT_20210715.pkl'))
        fault_name = rf_fault_model.predict(self.common_features_df)[0]

        # The model returns the string name.  We need the index to get a probability
        fault_id = rf_fault_model.classes_.tolist().index(fault_name)

        # predict_proba returns a mildly complicated np.array structure for our purposes different than documented.
        # It contains an array of predicted probabilities for each category indexed input example, then on classes_.
        fault_confidence = rf_fault_model.predict_proba(self.common_features_df)[0][fault_id]

        return {'fault-label': fault_name, 'fault-confidence': fault_confidence}

    @staticmethod
    def assert_valid_cavity_number(cavity_number):
        """Throws an exception if the supplied integer is not a valid cavity number.

            Args:
                cavity_number (int): The cavity number to evaluate.

            Raises:
                TypeError: if cavity_number is not an int
                ValueError: if cavity_number is not in range [1,8]
        """

        # Check that we have a valid cavity number
        if not isinstance(cavity_number, int):
            raise TypeError("cavity_number must be of type int")
        if not (cavity_number <= 8 or cavity_number >= 1):
            raise ValueError("cavity_number must be within span of [1,8]")


def main():
    if len(sys.argv) == 1:
        # Print an error/help message
        print("Error: Requires a single argument - the path to an RF waveform event folder")
        exit(1)
    else:
        # Analyze the faults that were passed
        data = []
        # The first argument is the main of this python script
        sys.argv.pop(0)

        # Iterate over each path and analyze the event
        for path in sys.argv:
            mod = Model(path)

            # Determine the zone and timestamp of the event from the path.  If the path is poorly formatted, this will
            # raise an exception
            zone = None
            timestamp = None
            error = None
            try:
                (zone, timestamp) = utils.path_to_zone_and_timestamp(path)
            except Exception as ex:
                # ex = sys.exc_info()
                # ex[1] is the exception message
                error = "{}".format(repr(ex))

            if error is None:
                # Try to analyze the fault.  If any of the validation routines fail, they will raise an exception.
                try:
                    result = mod.analyze()
                    data.append(result)
                except Exception as ex:
                    # ex = sys.exc_info()

                    result = {
                        # ex[1] is the exception message
                        'error': r'{}'.format(repr(ex)),
                        'location': zone,
                        'timestamp': timestamp
                    }
                    data.append(result)
            else:
                # If we had trouble parsing the location/timestamp info, don't try to analyze the fault.
                data.append({'error': error, 'path': path})

        print(json.dumps({'data': data}))


if __name__ == "__main__":
    main()
