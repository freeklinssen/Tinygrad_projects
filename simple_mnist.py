import numpy as np
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'tinygrad'))

import numpy as np
from tinygrad.tinygrad.tinygrad.tensor import Tensor
from tinygrad.tinygrad.tinygrad.nn import BatchNorm2D
from tinygrad.tinygrad.extra.utils import get_parameters
from tinygrad.tinygrad.datasets import fetch_mnist
from tinygrad.tinygrad.extra.training import train, evaluate, sparse_categorical_crossentropy
import tinygrad.tinygrad.nn.optim as optim
from tinygrad.tinygrad.extra.augment import augment_img



class ConvBlock:
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
    for cweight, cbias in zip(self.weights, self.biases):
      x = x.pad2d(padding=[1,1,1,1]).conv2d(cweight).add(cbias).relu()
    
    x = self._bn(x)
    
    return x
  

if __name__ == "__main__":
  
  tensor = Tensor.uniform(5,5)
  print(tensor)
  print('okay')
  print('ee')
  
  lmbd = 0.00025
  lossfn = lambda out,y: sparse_categorical_crossentropy(out, y) + lmbd*(model.weight1.abs() + model.weight2.abs()).sum()