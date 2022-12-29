import numpy as np
from sklearn import datasets
from tinygrad.tinygrad.tensor import Tensor
from tinygrad.tinygrad.nn import BatchNorm2D
from tinygrad.extra.utils import get_parameters
from tinygrad.extra.training import sparse_categorical_crossentropy
import tinygrad.tinygrad.nn.optim as optim

small = False


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
    
    
def train(model, x_train, y_train, optimizer, steps, lossfn,  bs=32):
  
  losses, accuracies = [], []
  Tensor.training = True
  for step in range(steps):
    numbers = np.random.uniform(0, len(x_train), bs)
    x= Tensor(x_train[numbers])
    y = Tensor(y_train[numbers])
    
    out = model.forward(x)
    loss = lossfn(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss = loss.data
    out = np.argmax(out.data, axis=-1)
    accuracy = (out == y).mean()
    
    print('loss: %.2f,accuracy: %.2f' % (loss, accuracy))
    
    losses.append(loss)
    accuracies.append(accuracy)
    
  return losses, accuracies
    

def evaluate(model, x_test, y_test, lossfn, bs=32, max_size = 120 ):
  
  Tensor.training = False
  epochs = max_size//bs
  numbers = np.random(0, len(x_test), epochs*bs)
  x = Tensor(x_test[numbers])
  y = y_test[numbers]
  
  for i in range(epochs):
    predictions[i*bs:(i+1)*bs] = model.forward(x[i*bs:(i+1)*bs])
    
  predictions = np.argmax(predictions.data, axis=-1)
  accuracy = (predictions == y).mean()
  
  print('test loss: %.2f, test accuracy: %.2f' % accuracy)
  
  return None 
     

if __name__ == "__main__":
  (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
  x_train = x_train.reshape(-1, 28, 28).astype(np.uint8)
  x_test = x_test.reshape(-1, 28, 28).astype(np.uint8)
  y_train = y_train.astype(np.uint8)
  y_test = y_test.astype(np.uint8)
  
  lrts = [1e-4, 1e-5]
  epochs = [10, 10]
  batch_size = 32
  steps = 20
  
  model = large_model()
  
  if small:
    steps = 10
    model = small_model()
  
  lmbd = 0.00025
  lossfn = lambda out,y: sparse_categorical_crossentropy(out, y) + lmbd*(model.linear1.abs().sum() + model.linear2.abs().sum())
  
  for lrt, epoch in zip(lrts, epochs):
    optimizer = optim.Adam(model.parameters(), lr=lrt)
    for i in range(epoch):
      train(model, x_train, y_train, optimizer, steps=steps, lossfn=lossfn, bs=batch_size)
      evaluate(model, x_train, y_train, max_size=120)