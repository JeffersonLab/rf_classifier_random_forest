#!/bin/env bash

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
APP_DIR="${SCRIPT_DIR}/.."

# Activate the apps python environment.
source ${APP_DIR}/venv/bin/activate
if [[ "$?" != "0" ]] ; then
    echo "Error activating python virtual environment.  Exiting."
    exit 1
fi

# Run the app passing along all of the args
python3 ${APP_DIR}/lib/model.py "$@"

# Unload the python environment
deactivate
