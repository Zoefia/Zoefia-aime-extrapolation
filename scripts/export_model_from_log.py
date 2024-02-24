"This script helps to export the trained model from the log to the model folder"

import os
from argparse import ArgumentParser

from aime.utils import MODEL_PATH

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_folder