import argparse
from typing import NamedTuple
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

def modeling_and_prediction(preprocess_train_data_path, preprocess_test_data_path, model_path, mlpipeline_ui_metadata_path):
    
    print("entering the modeling script")
    model_path=_make_parent_dirs_and_return_path(model_path)
    print("our model path is " + model_path)
    mlpipeline_ui_metadata_path=_make_parent_dirs_and_return_path(mlpipeline_ui_metadata_path)
    print("our pipline path is " + mlpipeline_ui_metadata_path)
    
    with open(preprocess_train_data_path, 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    with open(preprocess_test_data_path, 'rb') as f:
        X_test, y_test = pickle.load(f)
    
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

    model.compile(optimizers.Adam(learning_rate=0.001), 
                  loss=SparseCategoricalCrossentropy(), 
                  metrics=SparseCategoricalAccuracy(name='accuracy'))

    model.fit(np.array(X_train), np.array(y_train), validation_split=0.1, epochs=1, batch_size=64) 

    test_loss, test_acc = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
    print("Test_loss: {}, Test_accuracy: {}".format(test_loss, test_acc))

    model.save(f'{model_path}/model.h5')

    # Step 2: Prediction
    model = load_model(f'{model_path}/model.h5')

    y_pred = np.argmax(model.predict(X_test), axis=-1)

    cm = confusion_matrix(y_test, y_pred)
    vocab = list(np.unique(y_test))

    data = [(vocab[target_index], vocab[predicted_index], count) for target_index, target_row in enumerate(cm) for predicted_index, count in enumerate(target_row)]

    df = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
    df[['target', 'predicted']] = df[['target', 'predicted']].astype(int).astype(str)

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

    with open(mlpipeline_ui_metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file)

    conf_m_result = namedtuple('conf_m_result', ['mlpipeline_ui_metadata'])
    print("script done!")
    return conf_m_result(json.dumps(metadata))

def _make_parent_dirs_and_return_path(file_path: str):
    print("making the directories")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return file_path

if __name__ == '__main__':
    print("entering main")
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess-train-data-path', type=str, required=True)
    parser.add_argument('--preprocess-test-data-path', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-paths', type=str, required=True)
    args = parser.parse_args()
    modeling_and_prediction(args.preprocess_train_data_path, args.preprocess_test_data_path, args.model_path, args.output_paths)
