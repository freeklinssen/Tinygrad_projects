import numpy as np
import os
import sys
sys.path.append(os.getcwd())


import numpy as np
from tinygrad.tinygrad.tensor import Tensor
from tinygrad.tinygrad.nn import BatchNorm2D
from tinygrad.extra.utils import get_parameters
from tinygrad.datasets import fetch_mnist
from tinygrad.extra.training import train, evaluate, sparse_categorical_crossentropy
import tinygrad.tinygrad.nn.optim as optim
from tinygrad.extra.augment import augment_img



class Convolution:
  def __init__(self, h, w, inp, filters=32, conv=3):
    self.h, self.w = h, w
    self.inp = inp
    #init weights
    self.weights = [Tensor.uniform(filters, inp if i==0 else filters, conv, conv) for i in range(3)]
    self.biases = [Tensor.uniform(1, filters, 1, 1) for i in range(3)]
    #init layers
    self._bn = BatchNorm2D(filters)

  def __call__(self, input):
    x = input.reshape(shape=(-1, self.inp, self.w, self.h))
    for weight, bias in zip(self.weights, self.biases):
      x = x.pad2d(padding=[1,1,1,1]).conv2d(weight).add(bias).relu()
    
    x = self._bn(x)
    
    return x


class small_model:
  def __init__(self):
    self.conv_layer = Convolution(28,28,1)
    self.linear1 = Tensor.uniform(32*4, 32)
    self.linear2 = Tensor.uniform(32, 10)
  
  def parameters(self):
    return get_parameters(self)
  
  def forward(self, x):
    x = self.conv_layer(x) 
    x = x.avg_pool2d(kernel_size = (14,14))
    x = x.dot(self.linear1).softmax()
    x = x.dot(self.linear2).softmax()
    
    return x
  
  
class large_model:
  def __init__(self):
    self.conv_layer = [Convolution(28,28,1), Convolution(28,28,32), Convolution(14,14,32)]
    self.linear1 = Tensor.uniform(32, 16)
    self.linear2 = Tensor.uniform(16, 10)
  
  def parameters(self):
    return get_parameters(self)
  
  def forward(self, x):
    x = self.conv_layer[0](x)
    x = self.conv_layer[1](x)
    x = x.avg_pool2d(kernel_size = (2,2))
    x = self.conv_layer[2](x)
    x = x.avg_pool2d(kernel_size = (14,14))
    x = x.dot(self.linear1).softmax()
    x = x.dot(self.linear2).softmax()
    
    return x
    
    
def train(x):
  return x

def validation(x):
  return x


     

if __name__ == "__main__":
  
  lrt = 
  epochs = [10, 10]
  
  
  model = large_model()
  lmbd = 0.00025
  lossfn = lambda out,y: sparse_categorical_crossentropy(out, y) + lmbd*(model.weight1.abs() + model.weight2.abs()).sum()