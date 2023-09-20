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
    parser = argparse.ArgumentParser(prog='Download load preprocess data', description='')
    parser.add_argument("--download-link", dest="download_link", type=str, required=True)
    parser.add_argument("--output-data", dest="output_data_path", type=_make_parent_dirs_and_return_path, required=True)
    args = parser.parse_args()

    # Check and create output data path
    if not os.path.exists(args.output_data_path):
        os.makedirs(args.output_data_path)

    # Step 1: Download Data
    wget.download(args.download_link.format(file='train'), f'{args.output_data_path}/train_csv.zip')
    wget.download(args.download_link.format(file='test'), f'{args.output_data_path}/test_csv.zip')
    
    with zipfile.ZipFile(f"{args.output_data_path}/train_csv.zip", "r") as zip_ref:
        zip_ref.extractall(args.output_data_path)
        
    with zipfile.ZipFile(f"{args.output_data_path}/test_csv.zip", "r") as zip_ref:
        zip_ref.extractall(args.output_data_path)

    # Step 2: Load Data
    train_data_path = os.path.join(args.output_data_path, 'train.csv')
    test_data_path = os.path.join(args.output_data_path, 'test.csv')

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    ntrain = train_df.shape[0]
    all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
    print("all_data size is : {}".format(all_data.shape))

    # Step 3: Preprocess Data
    all_data_X = all_data.drop('label', axis=1)
    all_data_y = all_data.label

    all_data_X = all_data_X.values.reshape(-1, 28, 28, 1)
    all_data_X = all_data_X / 255.0

    X = all_data_X[:ntrain].copy()
    y = all_data_y[:ntrain].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    output_data_file_path = os.path.join(args.output_data_path, 'output_data.pkl')
    with open(output_data_file_path, 'wb') as f:
        pickle.dump((X_train, y_train, X_test, y_test), f)
        
    print('Done!')
    return output_data_file_path


if __name__ == "__main__":
    main()
