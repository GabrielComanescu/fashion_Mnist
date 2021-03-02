import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#The function already provided from the fashion mnist github repo
def load_mnist(path, kind='train'):
  import os
  import gzip
  import numpy as np

  """Load MNIST data from `path`"""
  labels_path = os.path.join(path,
                             '%s-labels-idx1-ubyte.gz'
                             % kind)
  images_path = os.path.join(path,
                             '%s-images-idx3-ubyte.gz'
                             % kind)

  with gzip.open(labels_path, 'rb') as lbpath:
      labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                             offset=8)

  with gzip.open(images_path, 'rb') as imgpath:
      images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                             offset=16).reshape(len(labels), 784)

  return images, labels


#Prepare the train and test data
#The test data already provided didn't work for me so i split
    #the training data into training and test
    #there should be enough for both train and test in 60000 images
def prepare_data():
  torch.manual_seed = 42

  x, y = load_mnist('data/', kind='train')

  X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

  X_train = torch.FloatTensor(np.copy(X_train))
  X_test = torch.FloatTensor(np.copy(X_test))
  y_train = torch.LongTensor(np.copy(y_train))
  y_test = torch.LongTensor(np.copy(y_test))

  train_data = TensorDataset(X_train, y_train)
  test_data = TensorDataset(X_test, y_test)

  return train_data, test_data


def get_data():

  train_data, test_data = prepare_data()

  train_loader = DataLoader(train_data, batch_size=100,shuffle=True)
  test_loader = DataLoader(test_data, batch_size=100,shuffle=False)

  return train_loader, test_loader
