# rf_classifier_random_forest
An rf_classifier pluggable model for analyzing CEBAF C100 RF fault waveform data.  This model utilizes a random forest modeling approach.

## Installation instructions
This software was developed against Python 3.6.9 to match the version of CEBAF's and rf_classifier's Python interpreter.  To install into CEBAF's control system enviroments perform the following steps.

### Install rf_classifier
Make sure rf_classifier has been installed.  See the Admin Guide at https://jeffersonlab.github.io/rf_classifier for more details.

### Download this model
Clone the repository and checkout the desired version.  You can install this directly to rf_classified's model directory OR to a secondary location and link it back into rf_classifier's model's directory.  This example shows placing it directly in rf_classifier's models directory.

```tcsh
cd /path/to/rf_classifier/models
git clone https://github.com/JeffersonLab/rf_classifier_random_forest  random_forest_<version>
cd random_forest_<version>
git checkout <version>
```

### CEBAF Certified Installation
This section describes a specific installation procedure used by the SQAM.  The General Installation section below provides generic instructions for installing elsewhere.  This guide assumes you followed the download instructions above and placed the model in /tmp/random_forest.

#### Build Step
From the temporary location, run "build" target to generate the virtual environment and download any dependencies.
```tcsh
cd /tmp/random_forest
./setup-certified.bash build
```

#### Test Step
Run the test target to test the model.
```tcsh
./setup-certified.bash test
```

#### Copy/Link to Certified Area
Refer to the rf_classifier application instructions for how to install the model into the certified directory tree and create the needed symlinks.

#### Install Step
***After having copied the model into /usr/csite/certified/... as directed in the application instructions***, run the setup-certified.bash install command.  This will delete any unnecessary files for model execution.
```tcsh
./setup-certified.bash install
```

#### Creating Certified Tarball
Back in the temporary area, add your audit files to docs/audit/ and any updates to the docs/release-notes.html.  Run the compact target to remove everything that isn't needed to be stored.  Then create the tarball.

```tcsh
vi audit/diff<version>.txt
vi release-notes.html
cd /tmp/random_forest
./setup-certified.bash compact
cd ..
mv random_forest random_forest_<version>
tar -czf random_forest_<version>.tar.gz random_forest_<version>
```

### General Installation
These are instructions for how to install this model into a standalone version of rf_classifier that is not part of the Certified Software repository and does not have to contend with supporting multiple architectures.

#### Download and Unzip
Clone the repository into the model directory of your local copy of rf_classifier and checkout the desired version.

```tcsh
git clone https://github.com/JeffersonLab/rf_classifier_random_forest rf_classifier/models/random_forest_v<version>
cd rf_classifier/models/random_forest_v<version>
git tag -l
git checkout v<version>
```

#### Setup Virtual Environment
The model was developed using Python 3.6.x.  Create a virtual environment based on that and install the requriements.
```tcsh
/usr/csite/pubtools/python/3.6/bin/python3 -m venv ./venv
source venv/bin/activate.csh
pip3 install -r requirements.txt
```

#### Testing
You can directly run tests for this model by running the following command.
```bash
python3 -m  test\test_model.py
```

This model project includes a test script (test_model.py) and data (test-data) that can be used to validate proper 
functioning of the model after installation.  See the rf_classifier Admin Guide at 
https://jeffersonlab.github.io/rf_classifier for additional documentation.  The short hand version is that you should run 
rf_classifier/tests/tester.bash to run all tests associated with all models.
