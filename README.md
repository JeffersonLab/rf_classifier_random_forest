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
git clone https://github.com/JeffersonLab/rf_classifier_random_forest  random_forest_\<version\>
cd random_forest_\<version\>
git checkout \<version\>
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

#### Certified Installation and Link Creation
Now copy the software to the versioned and architecture directory under /cs/certified/apps/rf_classifier/models/random_forest/<version>/<arch>/.  Afterwards, create a link for /usr/csite/certified/libexec/rf_classifier_models.  The link should be of the format <model_name>_<model_version>.

For example, if I were copying to version 0.1 on RHEL 7.
```tcsh
mkdir -p /cs/certified/apps/rf_classifier/models/random_forest/0.1/rhel-7-x86_64/
cp -R /tmp/random_forest/* /cs/certified/apps/rf_classifier/models/random_forest/0.1/rhel-7-x86_64/

mkdir -p /usr/csite/certified/libexec/rf_classifier_models/
cd /usr/csite/certified/libexec/rf_classifier_models/
ln -s /cs/certified/apps/rf_classifier/models/random_forest/0.1/rhel-7-x86_64 random_forest_v0_1
```

You may want to run the "install" target which will erase files and directories not used during execution.
```
cd /cs/certified/apps/rf_classifier/models/random_forest/<version>/<arch>
./setup-certified.bash install
```

#### Installing for rf_classifier Use
Each individual version of rf_classifier must be configured to use this new model as desired.  Simply create a link from that version of rf_classifier into the /usr/csite's libexec area.
```
cd /cs/certified/apps/rf_classifier/<version>/models
ln -s /usr/csite/certified/libexec/rf_classifier_models/random_forest_v<version> .
```

#### Creating Certified Tarball
Back in the temporary area, run the compact target to remove everything that isn't needed to be stored.  Then create the tarball.

```tcsh
cd /tmp/random_forest
./setup-certified.bash compact
cd ..
mv random_forest random_forest_<version>
tar -czf random_forest_<version>.tar.gz random_forest_<version>
```

### General Installation
This section assumes that you have downloaded the model directly into rf_classifier's models directory.

#### Create a python virtual envrironment and install package dependencies
```tcsh
/usr/csite/pubtools/python/3.6/bin/python3 -m venv ./venv
source venv/bin/activate.csh
pip3 install -r requirements.txt
```

#### Testing

This model project includes a test script (test_model.py) and data (test-data) that can be used to validate proper 
functioning of the model after installation.  See the rf_classifier Admin Guide at 
https://jeffersonlab.github.io/rf_classifier for official documentation.  The short hand version is that you should run 
rf_classifier/tests/tester.bash to run all tests associated with all models.  You can directly run tests for this model 
by running

```bash
python3 -m  test\test_model.py
```
