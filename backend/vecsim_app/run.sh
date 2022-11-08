#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo changing dir to ${SCRIPT_DIR}
cd ${SCRIPT_DIR}
echo $(pwd)

cd ../
pip install -e .
cd vecsim_app
python main.py
