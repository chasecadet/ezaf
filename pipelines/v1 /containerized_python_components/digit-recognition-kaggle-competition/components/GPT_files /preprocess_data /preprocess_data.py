# preprocess data

def preprocess_data(load_data_path: InputPath(str), 
                    preprocess_data_path: OutputPath(str)):
    
    # import Library
    import sys, subprocess;
    subprocess.run(["python", "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.run([sys.executable, '-m', 'pip', 'install','pandas'])
    subprocess.run([sys.executable, '-m', 'pip', 'install','scikit-learn'])
    import os, pickle;
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    #loading the train data
    with open(f'{load_data_path}/all_data', 'rb') as f:
        ntrain, all_data = pickle.load(f)
    
    # split features and label
    all_data_X = all_data.drop('label', axis=1)
    all_data_y = all_data.label
    
    # Reshape image in 3 dimensions (height = 28px, width = 28px , channel = 1)
    all_data_X = all_data_X.values.reshape(-1,28,28,1)

    # Normalize the data
    all_data_X = all_data_X / 255.0
    
    #Get the new dataset
    X = all_data_X[:ntrain].copy()
    y = all_data_y[:ntrain].copy()
    
    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    #creating the preprocess directory
    os.makedirs(preprocess_data_path, exist_ok = True)
    
    #Save the train_data as a pickle file to be used by the modelling component.
    with open(f'{preprocess_data_path}/train', 'wb') as f:
        pickle.dump((X_train,  y_train), f)
        
    #Save the test_data as a pickle file to be used by the predict component.
    with open(f'{preprocess_data_path}/test', 'wb') as f:
        pickle.dump((X_test,  y_test), f)
    
    return(print('Done!'))