def modeling(preprocess_data_path: InputPath(str), 
            model_path: OutputPath(str)):
    
    # import Library
    import sys, subprocess;
    subprocess.run(["python", "-m", "pip", "install", "--upgrade", "pip"])
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