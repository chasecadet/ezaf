def load_data(data_path: InputPath(str), 
              load_data_path: OutputPath(str)):
    
    # import Library
    import sys, subprocess;
    subprocess.run(["python", "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.run([sys.executable, '-m', 'pip', 'install','pandas'])
    # import Library
    import os, pickle;
    import pandas as pd
    import numpy as np

    #importing the data
    # Data Path
    train_data_path = data_path + '/train.csv'
    test_data_path = data_path + '/test.csv'

    # Loading dataset into pandas 
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    
    # join train and test together
    ntrain = train_df.shape[0]
    ntest = test_df.shape[0]
    all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
    print("all_data size is : {}".format(all_data.shape))
    
    #creating the preprocess directory
    os.makedirs(load_data_path, exist_ok = True)
    
    #Save the combined_data as a pickle file to be used by the preprocess component.
    with open(f'{load_data_path}/all_data', 'wb') as f:
        pickle.dump((ntrain, all_data), f)
    
    return(print('Done!'))