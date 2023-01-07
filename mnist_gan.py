import matplotlib.pyplot as plt
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
    self.linear4 = Tensor.uniform(1024, w*h)
  
  def parameters(self):
    return get_parameters(self)

  def forward(self, noise):
    x = noise.dot(self.linear1).leakyrelu(0.2)
    x = x.dot(self.linear2).leakyrelu(0.2)
    x = x.dot(self.linear3).leakyrelu(0.2)
    x = x.dot(self.linear4).tanh()
    x= x.reshape(shape=(-1, self.w, self.h))
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


def binary_crossentropy(out, y, BS):
  out = out.clip(1e-4, 1-1e-4)
  tmp = Tensor(np.ones((BS,1), np.float32), requires_grad=False)
  term_0 = tmp.sub(out).add(1e-4).log().mul(tmp.sub(y)).mean()
  term_1 = out.add(1e-4).log().mul(y).mean()
  result = term_0.add(term_1).mean().mul(-1)
  return result

    
def train(generator, discriminator, x_train, optimizer_g, optimizer_d, epoch, steps=100, BS=16):
  
  accuracies_discriminator, losses_discriminator = [], []
  accuracies_generator, losses_generator = [], []
  
    
  Tensor.training = True
  y_real = Tensor(np.ones(BS, np.float32))  # dit kan mischien nog anders i.e. andere target waardes 
  y_fake = Tensor(np.zeros(BS, np.float32))
  
  
  for step in range(steps):
    for i in range(4):
      ############# train discriminator:
      #real input:
      numbers = np.random.randint(0, len(x_train), BS)
      x_real = Tensor(x_train[numbers])
      
      output_real = discriminator.forward(x_real)
      loss_real = binary_crossentropy(output_real, y_real, BS)
      optimizer_d.zero_grad()
      loss_real.backward()
      optimizer_d.step()
      loss_real = loss_real.data
      output_real = [0 if x <= 0.5 else 1 for x in output_real.data]
      accuracy_real = (output_real == y_real.data).mean()
      
      
      #fake input
      noise = Tensor(np.random.randn(BS, 100))
      x_fake = generator.forward(noise).detach()
      
      output_fake = discriminator.forward(x_fake)
      loss_fake = binary_crossentropy(output_fake, y_fake, BS)
      optimizer_d.zero_grad()
      loss_fake.backward()
      optimizer_d.step()
      loss_fake = loss_fake.data
      output_fake = [0 if x <= 0.5 else 1 for x in output_fake.data]
      accuracy_fake = (output_fake == y_fake.data).mean()
      
      accuracies_discriminator.append(np.mean([accuracy_real, accuracy_fake]))
      losses_discriminator.append(loss_real + loss_fake)
      
    
    ############ train generator:
    
    noise = Tensor(np.random.randn(BS, 100))
    x_fake = generator.forward(noise)
    output_fake = discriminator.forward(x_fake)
    loss_fake = binary_crossentropy(output_fake, y_real, BS)
    optimizer_g.zero_grad() 
    loss_fake.backward()
    optimizer_g.step()
    losses_generator.append(loss_fake.data)
    output_fake = [0 if x <= 0.5 else 1 for x in output_fake.data]
    accuracies_generator.append((output_fake == np.ones(BS)).mean())
    
    print(f"EPOCH {epoch}, STEP {step}: Generator accuracies: {accuracies_generator[-1]}, Discriminator accuracies: {accuracies_discriminator[-1]}")


  #print some output of the generator
  noise = Tensor(np.random.randn(25, 100))
  generated_images = generator.forward(noise).detach().data
  generated_images = generated_images * 0.5 + 0.5
  h,w = 5, 5
  fig, axs = plt.subplots(h, w)
  count = 0
  for i in range(h):
      for j in range(w):
          axs[i,j].imshow(generated_images[count], cmap='gray')
          axs[i,j].axis('off')
          count += 1
  fig.savefig("mnist_gan_images/%d.png" % epoch)
  plt.close()
  
  return (np.mean(accuracies_discriminator), np.mean(losses_discriminator)), (np.mean(accuracies_generator), np.mean(losses_generator)) 
    

if __name__ == "__main__":
  (x_train,_), (_, _) = keras.datasets.mnist.load_data()
  x_train = x_train.reshape(-1, 28, 28).astype(np.uint32)
  x_train = x_train/125.5 - 1
  
  generator = generator(28, 28)
  discriminator = discriminator(28, 28)
  
  epochs = 10
  steps = 100
  batch_size = 16
  
  optim_g = optim.Adam(generator.parameters(), lr=0.0002, b1=0.5) 
  optim_d = optim.Adam(discriminator.parameters(), lr=0.0002, b1=0.5)
  
  for epoch in range(epochs):
    train(generator, discriminator, x_train, optim_g, optim_d, epoch, steps=steps, BS=batch_size)