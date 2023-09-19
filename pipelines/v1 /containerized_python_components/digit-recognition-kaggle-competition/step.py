# download data step
def download_data(download_link: str, data_path: OutputPath(str)):
    import zipfile
    import sys, subprocess;
    subprocess.run([sys.executable, "-m", "pip", "install", "wget"])
    import wget
    import os

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # download files
    wget.download(download_link.format(file='train'), f'{data_path}/train_csv.zip')
    wget.download(download_link.format(file='test'), f'{data_path}/test_csv.zip')
    
    with zipfile.ZipFile(f"{data_path}/train_csv.zip","r") as zip_ref:
        zip_ref.extractall(data_path)
        
    with zipfile.ZipFile(f"{data_path}/test_csv.zip","r") as zip_ref:
        zip_ref.extractall(data_path)
    
    return(print('Done!'))


# load data

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


def modeling(preprocess_data_path: InputPath(str), 
            model_path: OutputPath(str)):
    
    # import Library
    import sys, subprocess;
    subprocess.run([sys.executable, '-m', 'pip', 'install','pandas<2.0.0'])
    subprocess.run([sys.executable, '-m', 'pip', 'install','tensorflow'])
    import os, pickle;
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras, optimizers
    from tensorflow.keras.metrics import SparseCategoricalAccuracy
    from tensorflow.keras.losses import SparseCategoricalCrossentropy
    from tensorflow.keras import layers
    #loading the train data
    with open(f'{preprocess_data_path}/train', 'rb') as f:
        train_data = pickle.load(f)
        
    # Separate the X_train from y_train.
    X_train, y_train = train_data
    
    #initializing the classifier model with its input, hidden and output layers
    hidden_dim1=56
    hidden_dim2=100
    DROPOUT=0.5
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = hidden_dim1, kernel_size = (5,5),padding = 'Same', 
                         activation ='relu'),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Conv2D(filters = hidden_dim2, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Conv2D(filters = hidden_dim2, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation = "softmax")
            ])

    model.build(input_shape=(None,28,28,1))
    
    #Compiling the classifier model with Adam optimizer
    model.compile(optimizers.Adam(learning_rate=0.001), 
              loss=SparseCategoricalCrossentropy(), 
              metrics=SparseCategoricalAccuracy(name='accuracy'))

    # model fitting
    history = model.fit(np.array(X_train), np.array(y_train),
              validation_split=.1, epochs=1, batch_size=64)
    
    #loading the X_test and y_test
    with open(f'{preprocess_data_path}/test', 'rb') as f:
        test_data = pickle.load(f)
    # Separate the X_test from y_test.
    X_test, y_test = test_data
    
    # Evaluate the model and print the results
    test_loss, test_acc = model.evaluate(np.array(X_test),  np.array(y_test), verbose=0)
    print("Test_loss: {}, Test_accuracy: {} ".format(test_loss,test_acc))
    
    #creating the preprocess directory
    os.makedirs(model_path, exist_ok = True)
      
    #saving the model
    model.save(f'{model_path}/model.h5')    
    
    
def prediction(model_path: InputPath(str), 
                preprocess_data_path: InputPath(str), 
                mlpipeline_ui_metadata_path: OutputPath(str)) -> NamedTuple('conf_m_result', [('mlpipeline_ui_metadata', 'UI_metadata')]):
    
    # import Library
    import sys, subprocess;
    subprocess.run(["python", "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.run([sys.executable, '-m', 'pip', 'install','scikit-learn'])
    subprocess.run([sys.executable, '-m', 'pip', 'install','pandas<2.0.0'])
    subprocess.run([sys.executable, '-m', 'pip', 'install','tensorflow'])
    import pickle, json;
    import pandas as  pd
    import numpy as np
    from collections import namedtuple
    from sklearn.metrics import confusion_matrix
    from tensorflow.keras.models import load_model

    #loading the X_test and y_test
    with open(f'{preprocess_data_path}/test', 'rb') as f:
        test_data = pickle.load(f)
    # Separate the X_test from y_test.
    X_test, y_test = test_data
    
    #loading the model
    model = load_model(f'{model_path}/model.h5')
    
    # prediction
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    vocab = list(np.unique(y_test))
    
    # confusion_matrix pair dataset 
    data = []
    for target_index, target_row in enumerate(cm):
        for predicted_index, count in enumerate(target_row):
            data.append((vocab[target_index], vocab[predicted_index], count))
    
    # convert confusion_matrix pair dataset to dataframe
    df = pd.DataFrame(data,columns=['target','predicted','count'])
    
    # change 'target', 'predicted' to integer strings
    df[['target', 'predicted']] = (df[['target', 'predicted']].astype(int)).astype(str)
    
    # create kubeflow metric metadata for UI
    metadata = {
        "outputs": [
            {
                "type": "confusion_matrix",
                "format": "csv",
                "schema": [
                    {
                        "name": "target",
                        "type": "CATEGORY"
                    },
                    {
                        "name": "predicted",
                        "type": "CATEGORY"
                    },
                    {
                        "name": "count",
                        "type": "NUMBER"
                    }
                ],
                "source": df.to_csv(header=False, index=False),
                "storage": "inline",
                "labels": [
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                ]
            }
        ]
    }
    
    with open(mlpipeline_ui_metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file)

    conf_m_result = namedtuple('conf_m_result', ['mlpipeline_ui_metadata'])
    
    return conf_m_result(json.dumps(metadata))