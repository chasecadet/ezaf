import os
import pickle
import numpy as np
import pandas as pd
import json
from collections import namedtuple
from sklearn.metrics import confusion_matrix
from tensorflow import keras, optimizers
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import load_model

def modeling_and_prediction(preprocess_data_path: str, 
                            model_path: str, 
                            mlpipeline_ui_metadata_path: str) -> NamedTuple('conf_m_result', [('mlpipeline_ui_metadata', 'UI_metadata')]):

    # Step 1: Modeling
    # Load train data
    with open(f'{preprocess_data_path}/train', 'rb') as f:
        train_data = pickle.load(f)
        
    # Separate the X_train from y_train.
    X_train, y_train = train_data
    
    # Initializing the model
    hidden_dim1 = 56
    hidden_dim2 = 100
    DROPOUT = 0.5
    model = keras.Sequential([
            keras.layers.Conv2D(filters=hidden_dim1, kernel_size=(5,5), padding='Same', activation='relu'),
            keras.layers.Dropout(DROPOUT),
            keras.layers.Conv2D(filters=hidden_dim2, kernel_size=(3,3), padding='Same', activation='relu'),
            keras.layers.Dropout(DROPOUT),
            keras.layers.Conv2D(filters=hidden_dim2, kernel_size=(3,3), padding='Same', activation='relu'),
            keras.layers.Dropout(DROPOUT),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation="softmax")
        ])

    model.build(input_shape=(None, 28, 28, 1))

    # Compile the model
    model.compile(optimizers.Adam(learning_rate=0.001), 
                  loss=SparseCategoricalCrossentropy(), 
                  metrics=SparseCategoricalAccuracy(name='accuracy'))

    # Fit the model
    model.fit(np.array(X_train), np.array(y_train), validation_split=0.1, epochs=1, batch_size=64)
    
    # Load test data
    with open(f'{preprocess_data_path}/test', 'rb') as f:
        test_data = pickle.load(f)
    
    # Separate X_test and y_test
    X_test, y_test = test_data
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
    print("Test_loss: {}, Test_accuracy: {}".format(test_loss, test_acc))
    
    # Save the model
    os.makedirs(model_path, exist_ok=True)
    model.save(f'{model_path}/model.h5')

    # Step 2: Prediction
    # Load the model
    model = load_model(f'{model_path}/model.h5')

    # Prediction
    y_pred = np.argmax(model.predict(X_test), axis=-1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    vocab = list(np.unique(y_test))

    # Process confusion matrix data
    data = [(vocab[target_index], vocab[predicted_index], count) for target_index, target_row in enumerate(cm) for predicted_index, count in enumerate(target_row)]
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
    df[['target', 'predicted']] = df[['target', 'predicted']].astype(int).astype(str)
    
    # Create metadata
    metadata = {
        "outputs": [
            {
                "type": "confusion_matrix",
                "format": "csv",
                "schema": [
                    {"name": "target", "type": "CATEGORY"},
                    {"name": "predicted", "type": "CATEGORY"},
                    {"name": "count", "type": "NUMBER"}
                ],
                "source": df.to_csv(header=False, index=False),
                "storage": "inline",
                "labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            }
        ]
    }

    # Save metadata
    with open(mlpipeline_ui_metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file)

    conf_m_result = namedtuple('conf_m_result', ['mlpipeline_ui_metadata'])
    
    return conf_m_result(json.dumps(metadata))
