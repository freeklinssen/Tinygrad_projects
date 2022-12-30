import numpy as np
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'test'))

from tensorflow import keras
from tinygrad.tensor import Tensor
from tinygrad.nn import BatchNorm2D
from extra.utils import get_parameters
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
  
  def parameters(self):
    return get_parameters(self)

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
  
  def parameters(self):
    return get_parameters(self)
  
  def forward(self, x):
    x = x.reshape(shape=(-1, self.size))
    x = x.dot(self.linear1).leakyrelu(0.2)
    x = x.dot(self.linear2).leakyrelu(0.2)
    x = x.dot(self.linear3).sigmoid()
    return x
 
  
def binary_crossentropy(out, y):
  
  return None 
   
  
def train(generator, discriminator, x_train, lossfn, optimizer_g, optimizer_d, epoch, steps=100, BS = 32):
  
  Tensor.training = True
  y_real = np.ones((BS, 1))    # dit kan mischien nog anders i.e. andere target waardes
  y_fake = np.zeros((BS, 1))
    
  for step in steps:
    for i in range(4):
      ############# train discriminator:
      #real input:
      numbers = np.random.randint(0, len(x_train), BS)
      x_real = Tensor(x_train[numbers])
      
      output_real = discriminator.forward(x_real)
      loss_real = lossfn(output_real, y_real) # klopt niet
      optimizer_d.zero_grad()
      loss_real.backward()
      optimizer_d.step()
      loss_real = loss_real.data
      output_real = np.argmax(output_real.data, axis=-1)
      accuracy_real = (output_real == y_real).mean()
      
      
      #fake input
      noise = Tensor(np.random.normal(0, 1, (BS, 100)))
      x_fake = generator.forward(noise).detach()
      
      output_fake = discriminator.forward(x_fake)
      loss_fake = lossfn(output_fake, y_fake) # klopt niet
      optimizer_d.zero_grad()
      loss_fake.backward()
      optimizer_d.step()
      loss_fake = loss_fake.data
      output_fake = np.argmax(output_fake.data, axis=-1)
      accuracy_fake = (output_fake == y_real).mean()
      
      accuracy_discriminator = np.mean(accuracy_real, accuracy_fake)
      loss_discriminator = loss_real + loss_fake
      #total_predictions = np.concatenate((, accuracy_fake), axis=-1) 
      
    
    ############ train generator:
    
    noise = Tensor(np.random.normal(0, 1, (BS, 100)))
    x_fake = generator.forward(noise)
    output_fake = discriminator.forward(x_fake)
    loss_fake = lossfn(output_fake, y_real)
    optimizer_g.zero_grad()
    loss_fake.backward()
    optimizer_g.step()
    loss_generator = loss_fake.data
    output_fake = np.argmax(output_fake.data, axis=-1)
    accuracy_generator = (output_fake == y_real).mean()


  #print some output of the generator
  noise = Tensor(np.random.normal(0, 1, (25, 100)))
  generated_images = generator.forward(noise).detach()
  h,w = 5, 5
  fig, axs = plt.subplots(r, c)
  count = 0
  for i in range(h):
      for j in range(w):
          axs[i,j].imshow(generated_images[count, :,:,0], cmap='gray')
          axs[i,j].axis('off')
          count += 1
  fig.savefig("images/%d.png" % epoch)
  
  return  
    
    
    

if __name__ == "__main__":
  
  (x_train,_), (_, _) = keras.datasets.mnist.load_data()
  x_train = x_train.reshape(-1, 28, 28).astype(np.uint32)
  
  generator = generator(28, 28)
  discriminator = discriminator(28, 28)
  
  epochs = 10
  steps = 10
  batch_size = 32
  loss_function = lambda out, y: binary_crossentropy(out, y)
  
  optim_g = optim.Adam(generator.parameters(), lr=0.0002, b1=0.5) 
  optim_d = optim.Adam(discriminator.parameters(), lr=0.0002, b1=0.5)
  
  for epoch in range(epochs):
    train(generator, discriminator, x_train, loss_function, optim_g, optim_d, epoch, BS=batch_size)