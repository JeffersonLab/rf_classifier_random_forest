import os
import sys
import pandas as pd
import numpy as np
import math
import sklearn
import utils
import json

from statsmodels.tsa.ar_model import AR
from sklearn import preprocessing
from sklearn.externals import joblib

from base_model import BaseModel

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
        """Create a Model object  This only creates one additional member over BaseModel, the reduced_df DataFrame"""
        self.reduced_df = None
        self.common_features_df = None
        super().__init__(path)

    def analyze(self, deployment='ops'):
        """A method for analyzing the data held in event_dir that returns cavity and fault type label information.

        This method validates that the capture files and waveform data in event_dir are in the expected format and
        internally consistent.  First the cavity label is determined by the cavity model.  Should this return a
        "multiple" cavity label, the no fault type label determination is made.  Instead, a fault type label of
        "Multi Cav Turn Off" with the same confidence as the cavity label.
        """

        # Check that the data we're about to analyze meets any preconditions for our model
        self.validate_data(deployment)

        # Load the data from disk and parse it into a convenient pandas dataframe.
        self.parse_event_dir()

        # Fault and cavity models use same data and features.  Get that now.
        self.get_reduced_data()
        self.extract_features()

        # Analyze the data to determine which cavity caused the fault.
        cav_results = self.get_cavity_label()

        # A value of cavity-label '0' corresponds to a multi-cavity event.  In this case the fault analysis is
        # unreliable and we should short circuit and report only a multi-cavity fault type (likely someone performing
        # a zone wide operation triggering a "fault").  Use the cavity confidence since it is the prediction we're
        # basing this on.
        fault_results = {'fault-label': 'Multi Cav turn off', 'fault-confidence': cav_results['cavity-confidence']}
        if cav_results['cavity-label'] != 'multiple':
            fault_results = self.get_fault_type_label(int(cav_results['cavity-label']))

        # Get the name and timestamp of the event
        (zone, timestamp) = utils.path_to_zone_and_timestamp(self.event_dir)

        return {
            'location': zone,
            'timestamp': timestamp,
            'cavity-label': cav_results['cavity-label'],
            'cavity-confidence': cav_results['cavity-confidence'],
            'fault-label': fault_results['fault-label'],
            'fault-confidence': fault_results['fault-confidence']
        }

    def get_reduced_data(self):
        # Need to reduce the number of waveforms in order to reduce the computational complexity of feature extraction.
        # These were selected since this is what SRF experts use to "manually" determine which cavity and fault info.
        waveforms = [
            "1_GMES", "1_GASK", "1_CRFP", "1_DETA2",
            "2_GMES", "2_GASK", "2_CRFP", "2_DETA2",
            "3_GMES", "3_GASK", "3_CRFP", "3_DETA2",
            "4_GMES", "4_GASK", "4_CRFP", "4_DETA2",
            "5_GMES", "5_GASK", "5_CRFP", "5_DETA2",
            "6_GMES", "6_GASK", "6_CRFP", "6_DETA2",
            "7_GMES", "7_GASK", "7_CRFP", "7_DETA2",
            "8_GMES", "8_GASK", "8_CRFP", "8_DETA2"
        ]

        # Subset the event DataFrame to contain only the needed waveforms.
        self.reduced_df = self.event_df.loc[:, waveforms]

        # This is probably an unnecessary check, but it's good to be safe.
        if self.reduced_df.shape != (8192, len(waveforms)):
            raise ValueError("Cavity label dataset has improper dimensions.  Expected (8192," +
                             str(len(waveforms)) + ") received " + ascii(self.reduced_df.shape))

    def extract_features(self):
        """Computes the extracted features, common_features_df, for both models based on the reduced DataFrame.

            Returns None:  A DataFrame of cavity features, one per column.
        """

        # The standard scaler has issues with constant waveforms and sometimes returns +/-1 instead of zero.
        # This can cause exceptions in the follow on AR calls.
        signals = self.set_constant_waveforms_to_zero(self.reduced_df)

        # Standardized the signals to z-scores
        signal_scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
        signals = signal_scaler.fit_transform(signals)

        # Make a call to run auto-regression method and save the extracted features
        self.common_features_df = self.get_stat_AR_coefficients(signals, 5)

    def get_fault_type_label(self, cavity_number):
        """Determines the fault type based on a Model's reduced_df.  Examines the data for the specified cavity.

            Args:
                cavity_number (int): The number of the cavity (1-8) that caused the fault.

            Returns:
                dict:  A dictionary with format {'fault-label': <string_label>, 'fault-confidence': <float in [0,1]>}"
        """
        # Make sure we received a valid cavity number
        self.assert_valid_cavity_number(cavity_number)

        # Imputing on a single example is useless since there is no population to provide ranges or median values
        # Load the fault type model and make a prediction about the type of fault
        rf_fault_model = joblib.load(os.path.join(lib_dir, 'model_files', 'RF_FAULT_03112020.sav'))
        fault_id = rf_fault_model.predict(self.common_features_df)

        # predict_proba returns a mildly complicated np.array structure for our purposes different than documented.
        # It contains an array of predicted probabilities for each category indexed on classes_.
        # For some reason, accessing this value is returning as an array, so we need an extra [0] to get it as a number
        # as in the return statement.
        fault_confidence = rf_fault_model.predict_proba(self.common_features_df)[0][fault_id][0]

        # The fault type labels are encoded as numbers.  Need to create a LabelEncoder, load the encodings from disk
        # then "unencode" the fault_id to get the name of the fault type label.
        le = sklearn.preprocessing.LabelEncoder()

        # Default value of allow_pickle changed in newer versions.  Now =True is required here to allow loading of
        # objects arrays.
        le.classes_ = np.load(os.path.join(lib_dir, 'model_files', 'le_fault_classes.npy'), allow_pickle=True)
        fault_name = le.inverse_transform(fault_id)

        return {'fault-label': fault_name[0], 'fault-confidence': fault_confidence}

    @staticmethod
    def assert_valid_cavity_number(cavity_number):
        """Throws an exception if the supplied integer is not a valid cavity number.

            Raises:
                TypeError: if cavity_number is not an int
                ValueError: if cavity_number is not in range [1,8]
        """

        # Check that we have a valid cavity number
        if not isinstance(cavity_number, int):
            raise TypeError("cavity_number must be of type int")
        if not (cavity_number <= 8 or cavity_number >= 1):
            raise ValueError("cavity_number must be within span of [1,8]")

    @staticmethod
    def set_constant_waveforms_to_zero(df):
        """Set constant waveforms to all zero value.

        Expects a DataFrame with waveforms in columns.

            Args:
                df (DataFrame):  DataFrame of waveform values, one waveform per column

            Returns (DataFrame): A new DataFrame where the constant waveforms have been set to zero
        """
        out = df.copy()
        for column in out.columns:
            if np.max(out[column]) == np.min(out[column]):
                out[column].values[:] = 0

        return out

    def get_cavity_label(self):
        """Loads the underlying model and performs the predictions.

        This model operates on the common_features_df that is generated.

            Returns dictionary:  A dictionary containing both the cavity-label and cavity-confidence.
        """
        # Load the cavity model
        rf_cav_model = joblib.load(os.path.join(lib_dir, "model_files", 'RF_CAVITY_03112020.sav'))

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

    # noinspection PyPep8Naming
    def get_stat_AR_coefficients(self, signals, max_lag):
        """Get the auto-regression coefficients for a set of time series signals.

            Args:
                signals (DataFrame): A Pandas DataFrame of waveforms, one per column
                max_lag (float): The maximum number of AR coefficients to return.  Will be zero padded if model requires
                                  less than the number specified.

            Returns DataFrame: A dataframe that contains a single row where each column is a parameter coefficient.
        """
        for i in range(0, np.shape(signals)[1]):

            # The AR model throws for some constant signals.  The signals should have been normalized into z-scores, in
            # which case the parameters for an all zero signal are all zero.
            if self.is_constant_signal(signals[i]) and signals[0, i] == 0:
                parameters = np.append((np.zeros(max_lag + 1)))
            else:
                model = AR(signals[:, i])
                model_fit = model.fit(maxlag=max_lag, ic=None)
                if np.shape(model_fit.params)[0] < max_lag + 1:
                    parameters = np.pad(model_fit.params, (0, max_lag + 1 - np.shape(model_fit.params)[0]), 'constant',
                                        constant_values=0)
                elif np.shape(model_fit.params)[0] > max_lag + 1:
                    parameters = model_fit.params[: max_lag]
                else:
                    parameters = model_fit.params

            if i == 0:
                coefficients = parameters
            else:
                coefficients = np.append(coefficients, parameters, axis=0)

        return pd.DataFrame(coefficients).T

    def is_constant_signal(self, signal):
        """Is the supplied signal array only contains a single value, excluding NaN.  However, all NaNs returns true"""

        # Skip any NaNs
        start = 0
        length = len(signal)
        while start < length and math.isnan(signal[start]):
            start += 1

        # Only NaNs so return True (constant in some fashion)
        if start == length:
            return True

        # Work through the rest of the array, skipping NaNs
        prev = signal[start]
        for i in range(start + 1, len(signal)):
            if math.isnan(signal[i]):
                continue
            if prev != signal[i]:
                return False
            prev = signal[i]

        return True

    def validate_data(self, deployment='ops'):
        """Check that the event directory and it's data is of the expected format.

        This method inspects the event directory and raises an exception if a problem is found.  The following aspects
        of the event directory and waveform data are validated.
           # All eight cavities are represented by exactly one capture file
           # All of the required waveforms are represented exactly once
           # All of the capture files use the same timespan and have constant sampling intervals
           # All of the cavity are in the appropriate control mode (GDR I/Q => 4)

        Returns:
            None: Subroutines raise an exception if an error condition is found.

        """
        self.validate_capture_file_counts()
        self.validate_capture_file_waveforms()
        self.validate_waveform_times()
        self.validate_cavity_modes(deployment=deployment)
        self.validate_zones()


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
            except:
                ex = sys.exc_info()
                # ex[1] is the exception message
                error = "{}".format(ex[1])

            if error is None:
                # Try to analyze the fault.  If any of the validation routines fail, they will raise an exception.
                try:
                    result = mod.analyze()
                    data.append(result)
                except:
                    ex = sys.exc_info()

                    result = {
                        # ex[1] is the exception message
                        'error': r'{}'.format(ex[1]),
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
