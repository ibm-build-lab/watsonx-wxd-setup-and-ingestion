import argparse
import os
import uvicorn
import subprocess

# Run elastic/setup.py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", default="configs/ibm_config.yaml")
    return parser.parse_args()


args = parse_args()
subprocess.run(
    ["python3.10", "elastic/setup.py", "--config_file_path", args.config_file_path],
    check=True,
)
os.environ["CONFIG_FILE_PATH"] = args.config_file_path

uvicorn.run("app:app", host="0.0.0.0", port=8001)
