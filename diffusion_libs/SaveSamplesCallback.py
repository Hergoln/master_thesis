import tensorflow as tf
from tensorflow import keras
import numpy as np
from .dictionary_control import convert_back_to_code, fill_vocabulary
import os

def format_output(sample):
  return '\n'.join(str(element) for element in sample)

class SaveSamplesCallback(keras.callbacks.Callback):

  def __init__(self, path, samples_num, diffusion_steps, converter, scaler):
    super(SaveSamplesCallback, self).__init__()
    self.path = path
    self.samples_num = samples_num
    self.diffusion_steps = diffusion_steps
    self.converter = converter
    self.scaler = scaler
    fill_vocabulary()

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
    out = eval(f"f'{self.path}'")
    if not os.path.exists(out):
      return
    for counter in range(self.samples_num):
      epoch = epoch + 1
      with open(f"{out}\\model_result_code_{counter}.txt", 'w') as codeFileHandler, open(f"{out}\\model_result_tokens_{counter}.txt", 'w') as textFileHandler, open(f"{out}\\model_result_vals_{counter}.txt", 'w') as valFileHandler:
        for val in samples[counter]:
          valFileHandler.write(str(val) + '\n')
        for val in denormalized[counter]:
          textFileHandler.write(str(val) + '\n')
        scaled_up = self.scaler(denormalized[counter])
        converted = self.converter(scaled_up)
        codeFileHandler.write(format_output(converted))

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