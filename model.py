from ..base_model.base_model import BaseModel
import os


# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import glob

# For feature extraction
# from tsfresh import extract_features, extract_relevant_features, select_features
# from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters
# from tsfresh.utilities.dataframe_functions import impute
# from tsfresh.feature_extraction import settings

# For loading the trained model
# from sklearn.externals import joblib

# For translating the numerical result into a categroical label
# from sklearn import preprocessing


class Model(BaseModel):
    """
    This model is based on the first model produced by Chris Tennant.  It is based on a Random Forest machine learning model.

    Additional documentation is available in the docs folder.
    """

    @staticmethod
    def describe():
        return {
            'id': os.path.basename(os.path.dirname(__file__)),
            'releaseDate': "June 19, 2025",
            'cavLabels': ['cav1-8'],
            'faultLabels': ["Fault1", "Fault2", "SpecialFault", "none"],
            'brief': "Uses random forests for analysis",
            'details': Model.__doc__
        }

    @staticmethod
    def analyze():
        print("Doing random forest analysis")
