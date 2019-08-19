# Get the parent directory of this script
$scriptDir = split-path -parent $MyInvocation.MyCommand.Definition
$appDir = split-path -parent $scriptDir

# Activate the apps python environment.
& ${appDir}\venv\Scripts\Activate.ps1
if (! $?) {
    echo "Error activating python virtual environment.  Exiting."
    exit 1
}

# Run the app passing along all of the args
python.exe ${appDIR}/lib/model.py $args

deactivate