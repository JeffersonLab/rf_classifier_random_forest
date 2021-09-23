# rf_classifier_random_forest
An rf_classifier pluggable model for analyzing CEBAF C100 RF fault waveform data.  This model utilizes a random forest modeling approach, while input features may very from model to model.

This software was originaly developed against Python 3.6.9 to match the version of CEBAF's and rf_classifier's Python interpreter.  The most recent version uses Python 3.7.6 as to support more recent versions of some packages.

**Note:** Make sure rf_classifier has been installed prior to installing this model.  See the Admin Guide at 
https://jeffersonlab.github.io/rf_classifier for more details.  

## Documentation, Release Notes, etc.
https://jeffersonlab.github.io/rf_classifier_random_forest

## CEBAF Certified Installation
This section describes a specific installation procedure used by the SQAM.  The General Installation section below 
provides generic instructions for installing elsewhere.  Since this uses a virtual environment, it must be "built" in 
place and not copied from a temporary location.  This is because the virtual environment contain hardcoded absolute 
paths to their installation directory.

**Note: The following steps after "Downloading Source (Optional)" must be performed for each installation directory.** 

### rf_classifier Installation Instructions
Please start the installation procedure with the instructions from here http://devweb/controls_web/certified/rf_classifier/admin_guide/certified_install.html#sqam-model-install-guide.  This describes the process for linking in the new model with the existing certified app, and defers to this README for specific build/install/test steps related to this specific module.

### Download Source (Optional)
As the SQAM, the developer should have given you a code tarball to install.  If not, here is how you would download the
latest source from github. 
```tcsh
cd /tmp
git clone https://github.com/JeffersonLab/rf_classifier_random_forest  random_forest_<version>
cd random_forest_<version>
git tag -l
git checkout <version>
```

### Build Step
This model must be built in the location it will be installed in order for the virtual environment to function properly.
The installation directory is specified in the rf_classifier application certified installation guide.

For each install location, run "build" target to generate the virtual environment and download any dependencies.  Each
install location would have the same path, but on a different architecture.
```tcsh
cp -r /tmp/random_forest_<version> /path/to/certified/install/dir
cd /path/to/certified/install/dir
./setup-certified.bash build
```

### Test Step
Run the test target to test the model.
```tcsh
./setup-certified.bash test
```

### Install Step
Run the setup-certified.bash install command.  This will delete any unnecessary files for model execution.
```tcsh
./setup-certified.bash install
```

### Creating Certified Tarball
Back in the temporary area, add your audit files to docs/audit/ and any updates to the docs/release-notes.html.
Run the compact target to remove everything that isn't needed to be stored.  Then create the tarball.

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
These are instructions for how to install this model into a standalone version of rf_classifier that is not part of the
Certified Software repository and does not have to contend with supporting multiple architectures.

### Download and Unzip
Clone the repository into the model directory of your local copy of rf_classifier and checkout the desired version.

```tcsh
cd rf_classifier/models
git clone https://github.com/JeffersonLab/rf_classifier_random_forest random_forest_v<version>
cd random_forest_v<version>
git tag -l
git checkout v<version>
```

### App Setup
The setup_certified.bash script is the simplest way to perform a generic install.  Most users will simply want to run
the build and test targest.  These handle basic tasks like creating a virtual environment, uncompressing any large
files, and running unittests.  Note that the unit tests may require a JLab network connections.
```tcsh
./setup_certified.bash build
./setup_certified.bash test
```

Assuming all tests passed, the model should be safe to run from the bin directory.
