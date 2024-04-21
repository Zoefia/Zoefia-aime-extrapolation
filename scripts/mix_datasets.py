import os
from argparse import ArgumentParser

from aime.utils import DATA_PATH

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_dataset_names", type=str, nargs="+")
    parser.add_argument("-o", "--output_dataset_name", type=str, required=True)
    args = parser.parse_args()

    input_folders = [
        os.path.join(DATA_PATH, dataset_name) for dataset_name in