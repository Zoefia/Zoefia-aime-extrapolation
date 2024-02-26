"This script helps to export the trained model from the log to the model folder"

import os
from argparse import ArgumentParser

from aime.utils import MODEL_PATH

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_folder", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    log_folder = args.log_folder
    assert os.path.exists(log_folder)
    model_folder = os.path.join(MODEL_PATH, args.model_name)
    if not os.path.exists(model_folder):
        o