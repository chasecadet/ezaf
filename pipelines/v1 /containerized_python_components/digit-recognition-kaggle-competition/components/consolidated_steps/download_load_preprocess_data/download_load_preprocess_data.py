import os
import zipfile
import pickle
import wget
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import gc

def _make_parent_dirs_and_return_path(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return file_path

def validate_labels(df):
    assert df['label'].notna().all(), "NaN values found in labels"
    assert df['label'].between(0, 9).all(), "Invalid label values found"

def load_and_process_data(data_path, rows_at_a_time=50):
    col_names = ['label'] + [f'pixel_{i}' for i in range(1, 29)]
    with pd.read_csv(data_path, chunksize=rows_at_a_time, header=None, names=col_names) as reader:
        chunk_list = []
        for chunk in reader:
            try:
                labels = chunk['label']
                features = chunk.drop(columns=['label']).astype('float32')
                chunk = pd.concat([labels, features], axis=1)
                chunk_list.append(chunk)
            except Exception as e:
                print(f"An error occurred: {e}")
                continue
        
        data_df = pd.concat(chunk_list)
        del chunk_list
        gc.collect()
        
        return data_df


def main():
    print("oh here we goooooo") 
    parser = argparse.ArgumentParser(description='Download, load, and preprocess data')
    parser.add_argument("--download-link", dest="download_link", type=str, required=True)
    parser.add_argument("--output-train-data", dest="output_train_data_path", type=_make_parent_dirs_and_return_path, required=True)
    parser.add_argument("--output-test-data", dest="output_test_data_path", type=_make_parent_dirs_and_return_path, required=True)

    args = parser.parse_args()
    rows_at_a_time = 50
    download_link = args.download_link
    output_train_data_path = args.output_train_data_path
    output_test_data_path = args.output_test_data_path

    output_data_path = os.path.dirname(output_train_data_path)
    
    # Step 1: Download Data
    wget.download(download_link.format(file='train'), f'{output_data_path}/train_csv.zip')
    wget.download(download_link.format(file='test'), f'{output_data_path}/test_csv.zip')
    
    # Step 2: Extract and Load Data
    with zipfile.ZipFile(f"{output_data_path}/train_csv.zip", "r") as zip_ref:
        zip_ref.extractall(output_data_path)
    with zipfile.ZipFile(f"{output_data_path}/test_csv.zip", "r") as zip_ref:
        zip_ref.extractall(output_data_path)
    
    train_data_path = os.path.join(output_data_path, 'train.csv')
    test_data_path = os.path.join(output_data_path, 'test.csv')
    
    train_df = load_and_process_data(train_data_path, rows_at_a_time)
    test_df = load_and_process_data(test_data_path, rows_at_a_time)
    
    validate_labels(train_df)
    validate_labels(test_df)
    
    # Step 3: Preprocess Data
    ntrain = train_df.shape[0]
    all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
    del train_df, test_df
    gc.collect()
    
    validate_labels(all_data)
    
    all_data = all_data.sample(frac=.1, random_state=42).astype('float32')

    all_data_X = (all_data.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0).astype('float32')
    all_data_y = all_data.label

    X = all_data_X[:ntrain].copy()
    y = all_data_y[:ntrain].copy()

    del all_data, all_data_X, all_data_y
    gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42) 

    # Step 4: Save Data
    with open(output_train_data_path, 'wb') as f:
        pickle.dump((X_train, y_train), f)
    with open(output_test_data_path, 'wb') as f:
        pickle.dump((X_test, y_test), f)
    
    del X, y, X_train, X_test, y_train, y_test
    gc.collect()
    
    print('Done!')

if __name__ == "__main__":
    main()
