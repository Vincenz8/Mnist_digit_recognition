# 3rd party libraries
from PIL import Image
import requests
import zipfile
import numpy as np

# standard libraries 
from pathlib import Path
import os
import glob
import itertools

URL_DATASET = r"https://github.com/teavanist/MNIST-JPG/raw/master/MNIST%20Dataset%20JPG%20format.zip"

# folder's name inside zipfile
ZIP_FOLDER = "mnist.zip"
ZIP_ROOT = "MNIST Dataset JPG format"
ZIP_TRAIN = r"MNIST/MNIST - JPG - training"
ZIP_TEST = r"MNIST/MNIST - JPG - testing"

ROOT_DATA = Path("data")
DATATRAIN = r"data/MNIST/Training"
DATATEST = r"data/MNIST/Test"


def create_dataset(filename: str, img_shape: tuple[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return x, y

    Args:
        filename (str): folder containing images

    Returns:
        tuple[np.ndarray, np.ndarray]: x, y
    """
    grouped_images = {f"number{i}": glob.glob(f"{filename}/{i}/*") for i in range(10)}
    
    # get number of samples x class
    n_images = [len(grouped_images[k]) for k in grouped_images]
    
    # calculate cumulative sum for labels
    index = np.concatenate(([0], np.cumsum(n_images)))
    
    
    y = np.empty(shape=(index[-1],1), dtype=int)
    x = np.empty(shape=(index[-1], *img_shape), dtype=int) 
    
    all_images = itertools.chain.from_iterable(grouped_images.values()) 
      
    for i, img_path in enumerate(all_images):   
            x[i] = img_to_np(filename=img_path, img_shape= img_shape )
               
    for i in range(1, len(index)):
        y[index[i-1]:index[i]] = i-1
    
    return x, y

def img_to_np(filename: str, img_shape: tuple[int]) -> np.ndarray:
    with Image.open(filename) as img:
        arr_img = np.asarray(img)
        
    return arr_img.reshape(img_shape)
    
def download_dataset(url: str) -> None:
    """Download dataset

    Args:
        url (str): url
    """
    
    # download
    result = requests.get(url=url)
    with open("data/mnist.zip", "wb") as filezip:
        filezip.write(result.content)
       
def extract_rename(filename: str) -> None:
    """Extract and rename folder

    Args:
        filename (str): zipfile path
    """
    with zipfile.ZipFile(file= filename) as fz:
        fz.extractall(ROOT_DATA)
        
    os.rmdir(filename)
    
    os.rename(src=os.path.join(ROOT_DATA, ZIP_ROOT), dst= ROOT_DATA.joinpath("MNIST"))
    os.rename(src=ROOT_DATA.joinpath(ZIP_TEST), dst= DATATEST)
    os.rename(src=ROOT_DATA.joinpath(ZIP_TRAIN), dst= DATATRAIN)   
             
