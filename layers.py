from abc import ABC, abstractmethod
from re import L
from typing import List
import numpy as np
from sklearn.metrics import accuracy_score

import tqdm
from typing import List
from torchvision.datasets import MNIST

def weights_initialization(input_size, output_size):
    sigma = np.sqrt(2 / input_size)
    return np.random.normal(0, sigma, size=(input_size, output_size))
class Layer(ABC):

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_grad, learning_rate):
        raise NotImplementedError
class LinearLayer(Layer):
    
    def __init__(self, input_size, output_size):
        self.weights = weights_initialization(input_size, output_size)
        self.biases = np.zeros(output_size)
        self.data = None 
        
    def forward(self, x):
        self.x = x
        return x.dot(self.weights) + self.biases

    def backward(self, outpg, learning_rate):

      x_grad = self.x.reshape(-1, 1).dot(outpg.reshape(1, -1))
      self.weights = self.weights.copy()
      self.weights -= learning_rate * x_grad
      self.biases -= learning_rate * outpg
      return np.dot(outpg, np.transpose(self.weights))   

class ReLU:
  def __init__(self):
    self.data = None

  def forward(self, x):
    self.data = x.copy()
    x[x <= 0] = 0
    return x


  def backward(self, outpg, learning_rate) : 
    mask = (self.data >0).astype(np.float32)
    return mask * outpg

class SoftmaxCE:
  def __init__(self):
    self.data = None

  def forward(self, x):
    expx = np.exp(x - np.max(x))
    self.data = expx / expx.sum()
    return self.data

  def backward(self, outpg, learning_rate):
    self.data[outpg] -= 1
    return self.data

class Graph:

    def __init__(self, layers: List[Layer], learning_rate: float) -> None:
        self.layers = layers
        self.lr = learning_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y):
        for layer in reversed(self.layers):
            y = layer.backward(y, self.lr)
        return y

MNIST(".", download=True)