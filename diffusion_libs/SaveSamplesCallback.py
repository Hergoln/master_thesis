import tensorflow as tf
from tensorflow import keras
import numpy as np
from .dictionary_control import convert_back_to_code, fill_vocabulary
import os

def format_output(sample):
  return '\n'.join(str(element) for element in sample)

class SaveSamplesCallback(keras.callbacks.Callback):

  def __init__(self, path, samples_num, diffusion_steps, converter, scaler, history_path, append_history):
    super(SaveSamplesCallback, self).__init__()
    self.path = path
    self.samples_num = samples_num
    self.diffusion_steps = diffusion_steps
    self.converter = converter
    self.scaler = scaler
    self.history_path = history_path
    fill_vocabulary()
    self.is_history_file_created = False
    self.append_history = append_history

  def createFileHeaders(self, headers):
    if self.append_history:
      mode = "a"
    else:
      mode = "w"
    with open(self.history_path, mode) as historyFileHandler:
      historyFileHandler.write(','.join(headers) + '\n')

  def on_train_begin(self, logs=None):
    pass

  def on_train_end(self, logs=None):
    pass

  def on_epoch_begin(self, epoch, logs=None):
    pass

  def filesLine(self, model, samples, epoch_no):
    metrics = model.metrics
    toReturn = ''
    for m in metrics:
      toReturn += str(m.result().numpy()) + ','
    reduced =tf.math.reduce_mean(samples)
    return str(epoch_no) + ',' + toReturn + str(tf.math.abs(reduced).numpy().mean().mean()) + ',\n'

  def on_epoch_end(self, epoch, logs=None):
    headers = ['epoch', 'n_loss', 'i_loss', 'mean_values']
    if not self.is_history_file_created:
      self.createFileHeaders(headers)
      self.is_history_file_created = True
    samples, denormalized = self.model.generate(self.samples_num, self.diffusion_steps)
    samples = samples.numpy()
    denormalized = denormalized.numpy()
    epoch += 1
    out = eval(f"f'{self.path}'")
    with open(self.history_path, 'a') as historyFileHandler:
      historyFileHandler.write(self.filesLine(self.model, samples, epoch))
    if not os.path.exists(out):
      return
    for counter in range(self.samples_num):
      epoch = epoch + 1
      with (open(f"{out}\\model_result_code_{counter}.txt", 'w') as codeFileHandler, 
            open(f"{out}\\model_result_tokens_{counter}.txt", 'w') as textFileHandler, 
            open(f"{out}\\model_result_vals_{counter}.txt", 'w') as valFileHandler):
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