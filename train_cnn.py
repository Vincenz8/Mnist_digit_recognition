# 3rd party libraries 
import keras
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import pandas as pd

# local libraries
from data_utils import create_dataset, DATATRAIN, DATATEST

CNN = r"data/models/cnn.h5"
PERFORMANCE_CNN = r"data/models/performance.csv"

def main():
    # get data
    n_classes = 10
    img_shape = (28,28,1)
    
    x_train, y_train = create_dataset(filename=DATATRAIN, img_shape=img_shape)
    x_test, y_test = create_dataset(filename=DATATEST, img_shape=img_shape)
    
    
    # shuffling
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    
    # preprocessing
    encoder = OneHotEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train).toarray()
    y_test = encoder.transform(y_test).toarray()
    
    # init neural network
    architecture = [layers.Input(shape=img_shape),
                    layers.Conv2D(name="conv1", filters=64, kernel_size=5, activation="relu"),
                    layers.MaxPool2D(),
                    layers.Conv2D(name="conv2", filters=32, kernel_size=3, activation="relu"),
                    layers.MaxPool2D(),
                    layers.Flatten(),
                    layers.Dense(units=800, activation="relu"),
                    layers.Dense(units=n_classes, activation="softmax")]
    
    cnn = keras.Sequential(layers=architecture, name= "CNN")
    
    # print architecture
    cnn.summary()
    
    # training 
    cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = cnn.fit(x=x_train, y=y_train, batch_size= 64, epochs=20, validation_split=0.2)
    
    # evaluation
    print("\nEVALUATION")
    cnn.evaluate(x_test, y_test)
    
    # saving model and performance
    cnn.save(CNN)
    performance = pd.DataFrame(history.history)
    performance.to_csv(PERFORMANCE_CNN)
    
if __name__ == "__main__":
    main()