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

def main(download_link: str, output_data_path: str):
    print("oh here we goooooo") 
    _parser = argparse.ArgumentParser(prog='Download load preprocess data', description='')
    _parser.add_argument("--download-link", dest="download_link", type=str, required=True, default=argparse.SUPPRESS)
    _parser.add_argument("--output-data", dest="output_data_path", type=_make_parent_dirs_and_return_path, required=True, default=argparse.SUPPRESS)
    _parsed_args = vars(_parser.parse_args())
    _outputs = main(**_parsed_args)    
    # Step 1: Download Data
    os.makedirs(output_data_path, exist_ok=True)
    wget.download(download_link.format(file='train'), f'{output_data_path}/train_csv.zip')
    wget.download(download_link.format(file='test'), f'{output_data_path}/test_csv.zip')
    
    # Step 2: Extract and Load Data
    with zipfile.ZipFile(f"{output_data_path}/train_csv.zip", "r") as zip_ref:
        zip_ref.extractall(output_data_path)
    with zipfile.ZipFile(f"{output_data_path}/test_csv.zip", "r") as zip_ref:
        zip_ref.extractall(output_data_path)
    
    train_data_path = os.path.join(output_data_path, 'train.csv')
    test_data_path = os.path.join(output_data_path, 'test.csv')

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    # Step 3: Preprocess Data
    ntrain = train_df.shape[0]
    all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
    print("all_data size is : {}".format(all_data.shape))

    all_data_X = all_data.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0
    all_data_y = all_data.label

    X = all_data_X[:ntrain].copy()
    y = all_data_y[:ntrain].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Step 4: Save Data
    with open(f'{output_data_path}/train', 'wb') as f:
        pickle.dump((X_train, y_train), f)
    with open(f'{output_data_path}/test', 'wb') as f:
        pickle.dump((X_test, y_test), f)
    
    print('Done!')

if __name__ == "__main__":
    main()
