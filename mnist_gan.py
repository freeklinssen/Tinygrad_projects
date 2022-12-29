import numpy as np
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'test'))

from tensorflow import keras
from tinygrad.tensor import Tensor
from tinygrad.nn import BatchNorm2D
from extra.utils import get_parameters
from extra.training import sparse_categorical_crossentropy
import tinygrad.nn.optim as optim


class generator:
  def __init__(self, w, h, debt=1):
    self.h, self.w = w, h
    self.debt = debt
    self.linear1 = Tensor.uniform(100, 256)
    self.linear2 = Tensor.uniform(256, 512)
    self.linear3 = Tensor.uniform(512, 1024)
    self.linear4 = Tensor.uniform(1024, w*h*debt)
    
    self.bn = BatchNorm2D(1)

  def forward(self, noise):
    x = noise.dot(self.linear1)
    x = self.bn(x)
    x = noise.dot(self.linear2)
    x = self.bn(x)
    x = noise.dot(self.linear3)
    x = self.bn(x)
    x = noise.dot(self.linear4).tanh()
    x= x.reshape(shape=(-1, self.w, self.h, self.debt))
    return x


class discriminator:
  def __init__(self, w, h, depth=1):
    self.size = w*h*depth
    self.linear1 = Tensor.uniform(self.size, 512)
    self.linear2 = Tensor.uniform(512, 256)
    self.linear3 = Tensor.uniform(256, 1)
  
  def forward(self, x):
    x = x.reshape(shape=(-1, self.size))
    x = x.dot(self.linear1).relu()
    x = x.dot(self.linear2).sigmoid()
    return x
  

def train(x_train, epochs, BS = 32):
  
  half_BS = BS//2
  y_real = np.ones((BS, 1))
  y_fake = np.zeros((BS, 1))
  
  for epoch in range(epochs):
    
    ############# train discriminator:
    #real input:
    numbers = np.random.randint(0, len(x_train), BS)
    x_real = Tensor(x_train[numbers])
    
    #fake input
    noise = Tensor(np.random.normal(0, 1, (BS, 100)))
    x_fake = generator.forward(noise)
    
    P_y_real = discriminator.forward(x_real)
    P_y_fake = discriminator.forward(x_fake)
    
    # hier moet nog de loss berekent worden met de and backpropogated 
    
    ############ train generator:
    
    noise = Tensor(np.random.normal(0, 1, (BS, 100)))

    
    
    
    
  
  
  
    
    
    

if __name__ == "__main__":
  
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  x_train = x_train.reshape(-1, 28, 28).astype(np.uint8)
  x_test = x_test.reshape(-1, 28, 28).astype(np.uint8)
  y_train = y_train.astype(np.uint8)
  y_test = y_test.astype(np.uint8)
  
  generator = generator(28, 28)
  discriminator = discriminator(28, 28)
  
    
  print(np.ones((12, 1)))
  #self.noise = np.random.normal(0, 1, 100)
  #print(np.random.normal(0, 1, (25, 100)))
    
