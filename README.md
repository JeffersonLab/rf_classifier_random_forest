# rf_classifier_random_forest
An rf_classifier pluggable model for analyzing CEBAF C100 RF fault waveform data.  This model utilizes a random forest modeling approach.

## Installation instructions
This software was developed against Python 3.6.9 to match the version of CEBAF's and rf_classifier's Python interpreter.  To install into CEBAF's control system enviroments perform the following steps.

#### Install rf_classifier
Make sure rf_classifier has been installed.  See the Admin Guide at https://jeffersonlab.github.io/rf_classifier for more details.

#### Download this model
```tcsh
cd /path/to/rf_classifier/models
git clone https://github.com/JeffersonLab/rf_classifier_random_forest  random_forest_\<version\>
```

#### Create a python virtual envrironment and install package dependencies
```tcsh
/usr/csite/pubtools/python/3.6/bin/python3 -m venv ./venv
source venv/bin/activate.csh
pip3 install -r requirements.txt
```

#### Testing
This model project includes a test script (test_model.py) and data (test-data) that can be used to validate proper functioning of the model after installation.  See the rf_classifier Admin Guide at https://jeffersonlab.github.io/rf_classifier for official documentation.  The short hand version is that you should run rf_classifier/tests/tester.bash to run all tests associated with all models.
