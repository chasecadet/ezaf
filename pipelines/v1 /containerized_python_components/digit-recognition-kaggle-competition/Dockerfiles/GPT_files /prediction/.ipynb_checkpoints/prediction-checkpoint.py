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