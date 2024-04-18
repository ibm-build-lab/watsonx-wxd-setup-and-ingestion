export PYTHONPATH="$PWD:$PYTHONPATH"
input=${1:-configs/ibm_config.yaml}
python3 elastic/setup.py --config_file_path $input -s
# Ingest documents into elasticsearch
python3 elastic/ingest.py --config_file_path $input

