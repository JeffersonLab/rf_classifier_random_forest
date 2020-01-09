# rf_classifier_random_forest
An rf_classifier pluggable model for analyzing CEBAF C100 RF fault waveform data.  This model utilizes a random forest modeling approach.

This software was developed against Python 3.6.9 to match the version of CEBAF's and rf_classifier's Python interpreter.

**Note:** Make sure rf_classifier has been installed prior to installing this model.  See the Admin Guide at https://jeffersonlab.github.io/rf_classifier for more details.  

## Documentation, Release Notes, etc.
https://jeffersonlab.github.io/rf_classifier_random_forest

## CEBAF Certified Installation
This section describes a specific installation procedure used by the SQAM.  The General Installation section below 
provides generic instructions for installing elsewhere.  Since this uses a virtual environment, it must be "built" in 
place and not copied from a temporary location.  This is because the virtual environment contain hardcoded absolute 
paths to their installation directory.

**Note: The following steps after "Downloading The Model" must be performed for each installation location.** 

### Download The Model
Clone the repository and checkout the desired version to a temporary location.
```tcsh
cd /tmp
git clone https://github.com/JeffersonLab/rf_classifier_random_forest  random_forest_<version>
cd random_forest_<version>
git tag -l
git checkout <version>
```

### Build Step
For each install location, run "build" target to generate the virtual environment and download any dependencies.  Each
install location would have the same path, but on a different architecture.
```tcsh
cp -r /tmp/random_forest_<version> /usr/csite/certified/libexec/rf_classifier_models/

cd /usr/csite/certified/libexec/rf_classifier_models/random_forest_<version>
./setup-certified.bash build
```

### Test Step
Run the test target to test the model.
```tcsh
./setup-certified.bash test
```

### Copy/Link to Certified Area
Refer to the rf_classifier application instructions for how to install the model into the rf_classifier application and
create the needed symlinks.  This will only be done once per version of rf_classifier.

### Install Step
***After having copied the model into /usr/csite/certified/... as directed in the application instructions***, run the setup-certified.bash install command.  This will delete any unnecessary files for model execution.
```tcsh
./setup-certified.bash install
```

### Creating Certified Tarball
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

## General Installation
These are instructions for how to install this model into a standalone version of rf_classifier that is not part of the Certified Software repository and does not have to contend with supporting multiple architectures.

### Download and Unzip
Clone the repository into the model directory of your local copy of rf_classifier and checkout the desired version.

```tcsh
cd rf_classifier/models
git clone https://github.com/JeffersonLab/rf_classifier_random_forest random_forest_v<version>
cd random_forest_v<version>
git tag -l
git checkout v<version>
```

### Setup Virtual Environment
The model was developed using Python 3.6.x.  Create a virtual environment based on that and install the requriements.
```tcsh
/usr/csite/pubtools/python/3.6/bin/python3 -m venv ./venv
source venv/bin/activate.csh
pip3 install -r requirements.txt
```

### Testing
You can directly run tests for this model by running the following command.
```bash
python3 -m  test\test_model.py
```

This model project includes a test script (test_model.py) and data (test-data) that can be used to validate proper 
functioning of the model after installation.  See the rf_classifier Admin Guide at 
https://jeffersonlab.github.io/rf_classifier for additional documentation.  The short hand version is that you should run 
rf_classifier/tests/tester.bash to run all tests associated with all models.
