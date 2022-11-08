SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo changing dir to ${SCRIPT_DIR}
cd ${SCRIPT_DIR}
echo pwd


# uvicorn main:app --reload

#!/bin/sh

# python load_data.py
cd ../
pip install -e .
cd vecsim_app
python main.py
