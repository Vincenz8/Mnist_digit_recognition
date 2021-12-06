# 3rd party libraries 
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

# local libraries
from train_cnn import CNN, PERFORMANCE_CNN
from data_utils import create_dataset, DATATEST

    
def main():
   
    cnn = load_model(CNN)
    perf_cnn = pd.read_csv(PERFORMANCE_CNN)
    
    # get prediction
    x_test, y_test = create_dataset(filename=DATATEST, img_shape=(28,28,1))
    cnn_pred = np.argmax(cnn.predict(x_test), axis=1)
    
    # plot performance
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    
    fig.suptitle("Performance")
    # plot loss function
    ax1.set_title("Loss function")
    ax1.plot(perf_cnn["epoch"], perf_cnn["loss"], color="#c51f5d", label="Train Loss")
    ax1.plot(perf_cnn["epoch"], perf_cnn["val_loss"], color="#7bb3ff", label="Val Loss")
    ax1.legend()
    
    # plot accuracy
    ax2.set_title("Accuracy")
    ax2.plot(perf_cnn["epoch"], perf_cnn["accuracy"], color="#c51f5d", label="Train Accuracy")
    ax2.plot(perf_cnn["epoch"], perf_cnn["val_accuracy"], color="#7bb3ff", label="Val Accuracy")
    ax2.legend()
    
    fig.savefig("data/visualization/performance.png")
    
    # plot predictions
    indices = np.random.randint(x_test.shape[0], size=3)
    fig, axs = plt.subplots(nrows=1, ncols=3)
    fig.suptitle("Predictions")
    
    for ax, indices in zip(axs, indices):
        ax.imshow(x_test[indices])
        ax.set_title(F"Predicted: {cnn_pred[indices]}")
        ax.axis("off")
            
    fig.savefig("data/visualization/predictions.png")

if __name__ == "__main__":
    main()