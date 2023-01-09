import tensorflow as tf
from tensorflow import keras
import numpy as np

class CustomCallback(keras.callbacks.Callback):

  def __init__(self, path, samples_num, diffusion_steps):
    super(CustomCallback, self).__init__()
    self.path = path
    self.samples_num = samples_num
    self.diffusion_steps = diffusion_steps

  def on_train_begin(self, logs=None):
    pass

  def on_train_end(self, logs=None):
    pass

  def on_epoch_begin(self, epoch, logs=None):
    pass

  def on_epoch_end(self, epoch, logs=None):
    samples, denormalized = self.model.generate(self.samples_num, self.diffusion_steps)
    samples = samples.numpy()
    denormalized = denormalized.numpy()
    for counter in range(self.samples_num):
      epoch = epoch + 1
      out = eval(f"f'{self.path}'")
      with open(f"{out}_result_tokens_{counter}.txt", 'w') as textFileHandler, open(f"{out}_result_vals_{counter}.txt", 'w') as valFileHandler:
        for val in samples[counter]:
          valFileHandler.write(str(int(val)) + ' ')
        for val in denormalized[counter]:
          textFileHandler.write(str(val) + '\n')

  def on_test_begin(self, logs=None):
    pass

  def on_test_end(self, logs=None):
    pass

  def on_predict_begin(self, logs=None):
    pass

  def on_predict_end(self, logs=None):
    pass

  def on_train_batch_begin(self, batch, logs=None):
    pass

  def on_train_batch_end(self, batch, logs=None):
    pass

  def on_test_batch_begin(self, batch, logs=None):
    pass

  def on_test_batch_end(self, batch, logs=None):
    pass

  def on_predict_batch_begin(self, batch, logs=None):
    pass

  def on_predict_batch_end(self, batch, logs=None):
    pass