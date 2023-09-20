import os
import zipfile
import pickle
import wget
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

def _make_parent_dirs_and_return_path(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return file_path

def main():
    print("oh here we goooooo") 
    parser = argparse.ArgumentParser(description='Download load preprocess data')
    parser.add_argument("--download-link", dest="download_link", type=str, required=True)
    parser.add_argument("--output-train-data", dest="output_train_data_path", type=_make_parent_dirs_and_return_path, required=True)
    parser.add_argument("--output-test-data", dest="output_test_data_path", type=_make_parent_dirs_and_return_path, required=True)

    args = parser.parse_args()

    download_link = args.download_link
    output_train_data_path = args.output_train_data_path
    output_test_data_path = args.output_test_data_path

    print("our download link is " + download_link)

    output_data_path = os.path.dirname(output_train_data_path)
    
    # Step 1: Download Data
    os.makedirs(output_data_path, exist_ok=True)
    wget.download(download_link.format(file='train'), f'{output_data_path}/train_csv.zip')
    wget.download(download_link.format(file='test'), f'{output_data_path}/test_csv.zip')
    
    # Step 2: Extract and Load Data
    with zipfile.ZipFile(f"{output_data_path}/train_csv.zip", "r") as zip_ref:
        zip_ref.extractall(output_data_path)
    with zipfile.ZipFile(f"{output_data_path}/test_csv.zip", "r") as zip_ref:
        zip_ref.extractall(output_data_path)
    
    #... (keep the rest of your code here) ...

    # Step 4: Save Data
    with open(output_train_data_path, 'wb') as f:
        pickle.dump((X_train, y_train), f)
    with open(output_test_data_path, 'wb') as f:
        pickle.dump((X_test, y_test), f)
    
    print('Done!')

if __name__ == "__main__":
    main()

