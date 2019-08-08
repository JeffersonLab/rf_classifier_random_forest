import os
import pandas as pd
import numpy as np
import logging
import sklearn
import tsfresh
import utils
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings
from sklearn.externals import joblib
from base_model import BaseModel

lib_dir = os.path.join(os.path.dirname(__file__), 'lib')
"""The directory where pickle files containing tsfresh models, etc. can be found."""


class Model(BaseModel):
    """
    This model uses random forest models to identify the faulted cavity and fault type of a C100 event.

    This model is based on work done by Chris Tennant, Tom Powers, etc. and represents the initial model used to
    identify which cavity and fault type is associated with a C100 fault event.  Any individual cavity can be
    identified as the offending cavity.  Any collection of multiple cavities faulting at the same time are given the
    generic label of 'multiple'.  The following fault types may be identified by the model: E_Quench, Microphonics,
    Quench, 'Single Cav Turn off, and Multi-cav Turn Off.

    Additional documentation is available in the package docs folder.
    """

    @staticmethod
    def describe():
        """A method that provides descriptive information about the model."""
        return {
            'id': os.path.basename(os.path.dirname(__file__)),
            'releaseDate': "July 30, 2019",
            'cavLabels': ['multiple', '1', '2', '3', '4', '5', '6', '7', '8'],
            # TODO: verify that these are the correct fault labels
            'faultLabels': ['E_Quench', 'Microphonics', 'Quench', 'Single Cav Turn off', 'Multi Cav Turn off'],
            'brief': "Uses random forests ensemble method for analysis",
            'details': Model.__doc__
        }

    def analyze(self):
        """A method for analyzing the data held in event_dir that returns cavity and fault type label information.

        This method validates that the capture files and waveform data in event_dir are in the expected format and
        internally consistent.  First the cavity label is determined by the cavity model.  Should this return a
        "multiple" cavity label, the no fault type label determination is made.  Instead, a fault type label of
        "Multi Cav Turn Off" with the same confidence as the cavity label.
        """

        # Check that the data we're about to analyze meets any preconditions for our model
        self.validate_data()

        (zone, timestamp) = utils.path_to_zone_and_timestamp(self.event_dir)

        # Load the data from disk and parse it into a convenient pandas dataframe.
        self.parse_event_dir()

        # tsfresh requires an id column that allows for grouping related time series together for analysis.  Since
        # we want all of these time series to be treated as a single group, just supply a single id value (the '1' will
        # be repeated to match the index length.
        self.event_df.loc[:, 'id'] = pd.Series(1, index=self.event_df.index)

        # Analyze the data to determine which cavity caused the fault.
        cav_results = self.get_cavity_label()

        # A value of cavity-label '0' corresponds to a multi-cavity event.  In this case the fault analysis is
        # unreliable and we should short circuit and report only a multi-cavity fault type (likely someone performing
        # a zone wide operation triggering a "fault").  Use the cavity confidence since it is the prediction we're
        # basing this on.
        fault_results = {'fault-label': 'Multi Cav Turn off', 'fault-confidence': cav_results['cavity-confidence']}
        if cav_results['cavity-label'] != 'multiple':
            fault_results = self.get_fault_type_label(int(cav_results['cavity-label']))

        return {
            'location': zone,
            'timestamp': timestamp,
            'cavity-label': cav_results['cavity-label'],
            'cavity-confidence': cav_results['cavity-confidence'],
            'fault-label': fault_results['fault-label'],
            'fault-confidence': fault_results['fault-confidence']
        }

    def get_fault_type_label(self, cavity_number):
        """Determines the fault type based on a Model's event_df.  Examines the data for the specified cavity.

            Args:
                cavity_number (int): The number of the cavity (1-8) that caused the fault.

            Returns:
                dict:  A dictionary with format {'fault-label': <string_label>, 'fault-confidence': <float in [0,1]>}"

            Raises:
                TypeError: if cavity_number is not an int
                ValueError: if cavity_number is not in range [1,8]
        """

        # Check that we have a valid cavity number
        if not isinstance(cavity_number, int):
            raise TypeError("cavity_number must be of type int")
        if not (cavity_number <= 8 or cavity_number >= 1):
            raise ValueError("cavity_number must be within span of [1,8]")

        # Construct a dictionary for mapping cavity-specific waveform names to generic waveform names.  Used to
        # rename the dataframe columns into a generic set that can be analyzed by tsfresh regardless of which cavity
        # is the one to fault.
        waveforms = ('IMES', 'QMES', 'GMES', 'PMES', 'IASK', 'QASK', 'GASK', 'PASK', 'CRFP', 'CRFPP', 'CRRP', 'CRRPP',
                     'GLDE', 'PLDE', 'DETA2', 'CFQE2', 'DFQES')
        waveform_mapper = {str(cavity_number) + "_" + s: s for s in waveforms}
        fault_columns = list(waveform_mapper.keys())
        fault_columns.insert(0, 'id')
        fault_columns.insert(0, 'Time')

        # TODO: Change this so that fault_df is a copy of event_df, not a view.  This is probably changing the internal
        #   event_df even though you are changing fault_df.
        # Create a local subset of the event_df that has only the data needed for fault analysis
        fault_df = self.event_df.loc[:, fault_columns]

        # Rename waveforms to non-cavity specific names
        fault_df = fault_df.rename(columns=waveform_mapper)

        # Verify that we have a reasonable dataset to work on
        if fault_df.shape != (8192, len(fault_columns)):
            raise ValueError("Fault label dataset has improper dimensions.  Expected (8192," +
                             str(len(fault_columns)) + ") received " + ascii(fault_df.shape))

        # These are the top 50 features found from analyzing sklearn models - no way to generate this in a loop.
        top_features = [
            'GLDE__friedrich_coefficients__m_3__r_30__coeff_0',
            'PMES__max_langevin_fixed_point__m_3__r_30',
            'QMES__max_langevin_fixed_point__m_3__r_30',
            'GMES__fft_coefficient__coeff_16__attr_"real"',
            'DFQES__friedrich_coefficients__m_3__r_30__coeff_1',
            'GASK__max_langevin_fixed_point__m_3__r_30',
            'GASK__friedrich_coefficients__m_3__r_30__coeff_0',
            'GLDE__friedrich_coefficients__m_3__r_30__coeff_1',
            'GLDE__friedrich_coefficients__m_3__r_30__coeff_2',
            'PLDE__max_langevin_fixed_point__m_3__r_30',
            'GMES__friedrich_coefficients__m_3__r_30__coeff_0',
            'GMES__friedrich_coefficients__m_3__r_30__coeff_1',
            'PMES__friedrich_coefficients__m_3__r_30__coeff_0',
            'PMES__friedrich_coefficients__m_3__r_30__coeff_2',
            'GMES__fft_coefficient__coeff_49__attr_"angle"',
            'PMES__friedrich_coefficients__m_3__r_30__coeff_3',
            'GMES__fft_coefficient__coeff_49__attr_"real"',
            'PLDE__friedrich_coefficients__m_3__r_30__coeff_1',
            'PLDE__fft_coefficient__coeff_16__attr_"real"',
            'PLDE__friedrich_coefficients__m_3__r_30__coeff_0',
            'CRRP__fft_coefficient__coeff_75__attr_"angle"',
            'GMES__fft_coefficient__coeff_15__attr_"angle"',
            'QMES__agg_linear_trend__f_agg_"max"__chunk_len_50__attr_"intercept"',
            'DFQES__max_langevin_fixed_point__m_3__r_30',
            'GASK__fft_coefficient__coeff_24__attr_"abs"',
            'CRRP__max_langevin_fixed_point__m_3__r_30',
            'GMES__fft_coefficient__coeff_11__attr_"angle"',
            'PLDE__fft_coefficient__coeff_15__attr_"imag"',
            'GLDE__friedrich_coefficients__m_3__r_30__coeff_3',
            'CRRP__fft_coefficient__coeff_59__attr_"real"',
            'GMES__fft_coefficient__coeff_15__attr_"imag"',
            'GMES__fft_coefficient__coeff_44__attr_"angle"',
            'GASK__friedrich_coefficients__m_3__r_30__coeff_3',
            'IASK__fft_coefficient__coeff_29__attr_"abs"',
            'PLDE__fft_coefficient__coeff_32__attr_"real"',
            'PMES__friedrich_coefficients__m_3__r_30__coeff_1',
            'GMES__fft_coefficient__coeff_63__attr_"imag"',
            'PLDE__friedrich_coefficients__m_3__r_30__coeff_2',
            'QMES__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_3__w_2',
            'PLDE__fft_coefficient__coeff_30__attr_"imag"',
            'QMES__agg_linear_trend__f_agg_"min"__chunk_len_10__attr_"intercept"',
            'DFQES__friedrich_coefficients__m_3__r_30__coeff_0',
            'DFQES__friedrich_coefficients__m_3__r_30__coeff_3',
            'QMES__friedrich_coefficients__m_3__r_30__coeff_1',
            'GASK__friedrich_coefficients__m_3__r_30__coeff_2',
            'GMES__fft_coefficient__coeff_96__attr_"imag"',
            'GMES__fft_coefficient__coeff_30__attr_"imag"',
            'PLDE__fft_coefficient__coeff_80__attr_"abs"',
            'GMES__fft_coefficient__coeff_5__attr_"angle"',
            'PMES__energy_ratio_by_chunks__num_segments_10__segment_focus_4'
        ]

        # Setup the extraction settings that identifies which columns are to be extracted
        extraction_settings = settings.from_columns(top_features)

        # Extract the features needed for fault identification.  Since we're only dealing a single example, tsfresh
        # cannot effectively impute values for inf/None features.  But it does assign them a zero value and throws a
        # warning.  We want the zero, but don't want the warning so adjust the logging levels temporarily
        old_level = logging.getLogger('tsfresh').getEffectiveLevel()
        logging.getLogger('tsfresh').setLevel(logging.ERROR)
        fault_features = tsfresh.extract_features(fault_df.astype("float64"), column_id="id", column_sort="Time",
                                                  disable_progressbar=True,
                                                  impute_function=impute,
                                                  default_fc_parameters={},
                                                  kind_to_fc_parameters=extraction_settings,
                                                  show_warnings=False)
        logging.getLogger('tsfresh').setLevel(old_level)

        # Imputing on a single example is useless since there is no population to provide ranges or median values
        # Load the fault type model and make a prediction about the type of fault
        rf_fault_model = joblib.load(os.path.join(lib_dir, 'RF_fault_top50Features.pkl'))
        fault_id = rf_fault_model.predict(fault_features)

        # predict_proba returns a mildly complicated np.array structure for our purposes different than documented.
        # It contains an array of predicted probabilities for each category indexed on classes_.
        # For some reason, accessing this value is returning as an array, so we need an extra [0] to get it as a number
        # as in the return statement.
        fault_confidence = rf_fault_model.predict_proba(fault_features)[0][fault_id][0]

        # The fault type labels are encoded as numbers.  Need to create a LabelEncoder, load the encodings from disk
        # then "unencode" the fault_id to get the name of the fault type label.
        le = sklearn.preprocessing.LabelEncoder()
        le.classes_ = np.load(os.path.join(lib_dir, 'le_fault_classes.npy'), allow_pickle=True)
        fault_name = le.inverse_transform(fault_id)

        return {'fault-label': fault_name[0], 'fault-confidence': fault_confidence}

    def get_cavity_label(self):
        """Analyzes the Model's event_df to determine which cavity was responsible for the fault.

            Returns:
                dict:  A dictionary with format {'cavity-label': <string_label>, 'cavity-confidence': <float in [0,1]>}"
        """

        # Need to reduce the number of waveforms in order to reduce the computational complexity of feature extraction.
        # These were selected since this is what SRF experts use to "manually" determine which cavity faulted.
        cavity_label_waveforms = [
            "Time", "id",
            "1_GMES", "1_GASK", "1_CRFP", "1_DETA2", "2_GMES", "2_GASK", "2_CRFP", "2_DETA2",
            "3_GMES", "3_GASK", "3_CRFP", "3_DETA2", "4_GMES", "4_GASK", "4_CRFP", "4_DETA2",
            "5_GMES", "5_GASK", "5_CRFP", "5_DETA2", "6_GMES", "6_GASK", "6_CRFP", "6_DETA2",
            "7_GMES", "7_GASK", "7_CRFP", "7_DETA2", "8_GMES", "8_GASK", "8_CRFP", "8_DETA2"
        ]

        # Subset the event dataframe to contain only the needed waveforms.
        cavity_df = self.event_df.loc[:, cavity_label_waveforms]

        # This is probably an unnecessary check, but it's good to be safe
        if cavity_df.shape != (8192, len(cavity_label_waveforms)):
            raise ValueError("Cavity label dataset has improper dimensions.  Expected (8192," +
                             str(len(cavity_label_waveforms)) + ") received " + ascii(cavity_df.shape))

        # Setup the dictionary that specifies which features will be extracted from the data
        fft_only = {'fft_coefficient': []}
        for attr in ('real', 'imag', 'abs', 'angle'):
            for coeff in range(0, 100):  # produces [0, 1, ..., 99]
                fft_only['fft_coefficient'].append({'attr': attr, 'coeff': coeff})

        # Extract the features needed to identify which cavity faulted
        cavity_features = tsfresh.extract_features(cavity_df.astype('float64'),
                                                   disable_progressbar=True,
                                                   column_sort="Time",
                                                   column_id="id",
                                                   impute_function=impute,
                                                   default_fc_parameters=fft_only
                                                   )

        # Load the model from disk and make a prediction about which cavity faulted first.  The predict() method returns
        # an array of results.  We only have one result, so pull it out of the array structure now
        rf_cav_model = joblib.load(os.path.join(lib_dir, 'RandomForest_cavity.pkl'))
        cavity_id = rf_cav_model.predict(cavity_features)
        cavity_id = cavity_id[0]

        # predict_proba returns a mildly complicated 2D np.array structure.  First index is for each supplied example.
        # Second index is for the predicted probabilities for each category indexed on classes_.   We have one example,
        # and want to probability for the class corresponding to cavity_id (the actual prediction, i.e., the class with
        # the greatest probability).
        cavity_confidence = rf_cav_model.predict_proba(cavity_features)[0][cavity_id]

        # Convert the results from an int to a human readable string
        if cavity_id == 0:
            cavity_id = 'multiple'
        else:
            # The cavity_id int corresponds to the actual cavity number if 1-8
            cavity_id = str(cavity_id)

        return {'cavity-label': cavity_id, 'cavity-confidence': cavity_confidence}

    def validate_data(self):
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
        self.validate_cavity_modes()
