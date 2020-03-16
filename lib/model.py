import os
import sys
import pandas as pd
import numpy as np
import math
import logging
import sklearn
import tsfresh
import utils
import json

from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings
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

    def analyze(self, deployment='ops'):
        """A method for analyzing the data held in event_dir that returns cavity and fault type label information.

        This method validates that the capture files and waveform data in event_dir are in the expected format and
        internally consistent.  First the cavity label is determined by the cavity model.  Should this return a
        "multiple" cavity label, the no fault type label determination is made.  Instead, a fault type label of
        "Multi Cav Turn Off" with the same confidence as the cavity label.
        """

        # Check that the data we're about to analyze meets any preconditions for our model
        self.validate_data(deployment)

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
        fault_results = {'fault-label': 'Multi Cav turn off', 'fault-confidence': cav_results['cavity-confidence']}
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
        """

        self.assert_valid_cavity_number(cavity_number)
        fault_df = self.get_fault_data(cavity_number)
        fault_features = self.get_fault_features(fault_df)
        return self.get_fault_type_results(fault_features)

    def assert_valid_cavity_number(self, cavity_number):
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

    def get_fault_data(self, cavity_number):
        """Gets the fault data needed for the fault type prediction."""
        # Construct a dictionary for mapping cavity-specific waveform names to generic waveform names.  Used to
        # rename the DataFrame columns into a generic set that can be analyzed by tsfresh regardless of which cavity
        # is the one to fault.
        waveforms = ('IMES', 'QMES', 'GMES', 'PMES', 'IASK', 'QASK', 'GASK', 'PASK', 'CRFP', 'CRFPP', 'CRRP', 'CRRPP',
                     'GLDE', 'PLDE', 'DETA2', 'CFQE2', 'DFQES')
        waveform_mapper = {str(cavity_number) + "_" + s: s for s in waveforms}
        fault_columns = list(waveform_mapper.keys())
        fault_columns.insert(0, 'id')
        fault_columns.insert(0, 'Time')

        # Create a local subset of the event_df that has only the data needed for fault analysis
        fault_df = self.event_df.copy()
        fault_df = fault_df.loc[:, fault_columns]

        # Rename waveforms to non-cavity specific names
        fault_df = fault_df.rename(columns=waveform_mapper)

        # Verify that we have a reasonable dataset to work on
        if fault_df.shape != (8192, len(fault_columns)):
            raise ValueError("Fault label dataset has improper dimensions.  Expected (8192," +
                             str(len(fault_columns)) + ") received " + ascii(fault_df.shape))

        return fault_df

    def get_fault_features(self, fault_df):
        """Returns the extracted features needed for fault type prediction.

        Expects IMES, QMES, GMES, PMES, IASK, QASK, GASK, PASK, CRFP, CRFPP, CRRP CRRPP, GLDE, PLDE, DETA2, CFQE2, and
        DFQES signals for the faulted cavity.
        """

        # These are the top 50 features found from analyzing sklearn models - no way to generate this in a loop.
        top_features = ['GMES__augmented_dickey_fuller__attr_"usedlag"',
                        'CRFP__maximum',
                        'CRFP__agg_linear_trend__f_agg_"var"__chunk_len_50__attr_"stderr"',
                        'GASK__fft_coefficient__coeff_64__attr_"abs"',
                        'CRFP__augmented_dickey_fuller__attr_"teststat"',
                        'GMES__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.4',
                        'CRFP__augmented_dickey_fuller__attr_"pvalue"',
                        'QMES__augmented_dickey_fuller__attr_"usedlag"',
                        'GMES__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.6',
                        'PLDE__augmented_dickey_fuller__attr_"usedlag"',
                        'CRFP__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8',
                        'CRFP__fft_coefficient__coeff_96__attr_"abs"',
                        'GMES__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.6',
                        'IMES__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.4',
                        'PLDE__partial_autocorrelation__lag_5',
                        'PLDE__minimum',
                        'GMES__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.6',
                        'QMES__approximate_entropy__m_2__r_0.5',
                        'PLDE__max_langevin_fixed_point__m_3__r_30',
                        'QASK__fft_coefficient__coeff_17__attr_"abs"',
                        'CRFP__agg_linear_trend__f_agg_"var"__chunk_len_50__attr_"slope"',
                        'CRFP__fft_coefficient__coeff_48__attr_"abs"',
                        'GMES__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4',
                        'GMES__number_cwt_peaks__n_1',
                        'CRFP__ar_coefficient__k_10__coeff_0',
                        'QMES__approximate_entropy__m_2__r_0.7',
                        'GASK__fft_aggregated__aggtype_"skew"',
                        'CRRP__longest_strike_above_mean',
                        'PLDE__partial_autocorrelation__lag_6',
                        'IMES__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.4',
                        'PLDE__fft_coefficient__coeff_82__attr_"real"',
                        'CRFP__ratio_beyond_r_sigma__r_1.5',
                        'PLDE__friedrich_coefficients__m_3__r_30__coeff_0',
                        'IMES__fft_coefficient__coeff_11__attr_"abs"',
                        'GASK__ratio_beyond_r_sigma__r_6',
                        'GASK__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8',
                        'QASK__fft_aggregated__aggtype_"skew"',
                        'IMES__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4',
                        'IASK__fft_aggregated__aggtype_"skew"',
                        'DETA2__longest_strike_above_mean',
                        'CRRPP__quantile__q_0.9',
                        'GASK__fft_aggregated__aggtype_"centroid"',
                        'QASK__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.0',
                        'PLDE__partial_autocorrelation__lag_2',
                        'DFQES__ratio_value_number_to_time_series_length',
                        'PLDE__fft_coefficient__coeff_43__attr_"angle"',
                        'CRFP__percentage_of_reoccurring_datapoints_to_all_datapoints',
                        'CRFP__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.8',
                        'QASK__spkt_welch_density__coeff_5',
                        'PLDE__autocorrelation__lag_9']

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

        # Need to make sure that the extracted features are in the same order as for the training set.  This order
        # matches the order of the the top_features array.
        fault_features = fault_features[top_features]

        # The tsfresh features were standardized based on the training data set.  Load those mean and variances, and
        # standardized this data as that is what the model is expecting
        fault_mean = np.load(os.path.join(lib_dir, "model_files", "RF_FAULT_top50_mean.npy"))
        fault_var = np.load(os.path.join(lib_dir, "model_files", "RF_FAULT_top50_var.npy"))
        fault_features = (fault_features - fault_mean) / fault_var

        return fault_features

    def get_fault_type_results(self, fault_features):
        """Get the fault type label and the confidence associated with the given fault features."""
        # Imputing on a single example is useless since there is no population to provide ranges or median values
        # Load the fault type model and make a prediction about the type of fault
        rf_fault_model = joblib.load(os.path.join(lib_dir, 'model_files', 'RF_FAULT_top50_01292020.sav'))
        fault_id = rf_fault_model.predict(fault_features)

        # predict_proba returns a mildly complicated np.array structure for our purposes different than documented.
        # It contains an array of predicted probabilities for each category indexed on classes_.
        # For some reason, accessing this value is returning as an array, so we need an extra [0] to get it as a number
        # as in the return statement.
        fault_confidence = rf_fault_model.predict_proba(fault_features)[0][fault_id][0]

        # The fault type labels are encoded as numbers.  Need to create a LabelEncoder, load the encodings from disk
        # then "unencode" the fault_id to get the name of the fault type label.
        le = sklearn.preprocessing.LabelEncoder()

        # Default value of allow_pickle changed in newer versions.  Now =True is required here to allow loading of
        # objects arrays.
        le.classes_ = np.load(os.path.join(lib_dir, 'model_files', 'le_fault_classes.npy'), allow_pickle=True)
        fault_name = le.inverse_transform(fault_id)

        return {'fault-label': fault_name[0], 'fault-confidence': fault_confidence}

    def get_cavity_label(self):
        """Analyzes the Model's event_df to determine which cavity was responsible for the fault.

            Returns:
                dict:  A dictionary with format {'cavity-label': <string_label>, 'cavity-confidence': <float in [0,1]>}"
        """

        # Get the data needed to perform the cavity prediction
        cavity_df = self.get_cavity_data()

        # Extract the features from the waveform data
        cavity_features = self.get_cavity_features(cavity_df)

        # Get the results of the model
        cavity_results = self.get_cavity_results(cavity_features)

        return cavity_results

    def get_cavity_data(self):
        """Gets the subset of waveform data needed for the cavity label prediction"""

        # Need to reduce the number of waveforms in order to reduce the computational complexity of feature extraction.
        # These were selected since this is what SRF experts use to "manually" determine which cavity faulted.
        cavity_label_waveforms = [
            # "Time", "id",
            "1_GMES", "1_GASK", "1_CRFP", "1_CRFPP", "1_DETA2",
            "2_GMES", "2_GASK", "2_CRFP", "2_CRFPP", "2_DETA2",
            "3_GMES", "3_GASK", "3_CRFP", "3_CRFPP", "3_DETA2",
            "4_GMES", "4_GASK", "4_CRFP", "4_CRFPP", "4_DETA2",
            "5_GMES", "5_GASK", "5_CRFP", "5_CRFPP", "5_DETA2",
            "6_GMES", "6_GASK", "6_CRFP", "6_CRFPP", "6_DETA2",
            "7_GMES", "7_GASK", "7_CRFP", "7_CRFPP", "7_DETA2",
            "8_GMES", "8_GASK", "8_CRFP", "8_CRFPP", "8_DETA2"
        ]

        # Subset the event dataframe to contain only the needed waveforms.
        cavity_df = self.event_df.loc[:, cavity_label_waveforms]

        # This is probably an unnecessary check, but it's good to be safe.
        if cavity_df.shape != (8192, len(cavity_label_waveforms)):
            raise ValueError("Cavity label dataset has improper dimensions.  Expected (8192," +
                             str(len(cavity_label_waveforms)) + ") received " + ascii(cavity_df.shape))

        return cavity_df

    def get_cavity_features(self, cavity_df):
        """Computes the extracted features needed for the cavity model based on the supplied signals.

            Args:
                cavity_df (pd.DataFrame): A DataFrame with one waveform signal per column.  Does not expect an ID
                                            or time column, only cavity waveform values.

            Returns DataFrame:  A DataFrame of cavity features, one per column.
        """

        # The standard scaler has issues with constant waveforms and sometimes returns +/-1 instead of zero.
        # This can cause exceptions in the follow on AR calls.
        signals = self.set_constant_waveforms_to_zero(cavity_df)

        # Standardized the signals to z-scores
        signal_scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
        signals = signal_scaler.fit_transform(signals)

        # Make a call to run auto-regression method and return it's results
        return self.get_stat_AR_coefficients(signals, 5)

    def set_constant_waveforms_to_zero(self, df):
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

    def get_cavity_results(self, cavity_features):
        """Loads the underlying model and performs the predictions.

        This model expects AR features with max_lag=5, based on standardized GMES, GASK, CRFP, CRFPP, DETA2 from all
        eight cavities.

            Returns dictionary:  A dictionary containing both the cavity-label and cavity-confidence.
        """
        # Load the cavity model
        rf_cav_model = joblib.load(os.path.join(lib_dir, "model_files", 'RF_CAVITY_AR_02202020.sav'))

        # Load the model from disk and make a prediction about which cavity faulted first.  The predict() method returns
        # an array of results.  We only have one result, so pull it out of the array structure now
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
