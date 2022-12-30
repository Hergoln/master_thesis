from diffusion_libs import *
from tensorflow import keras

def main():
  # data
  num_epochs = 5 

  # sampling
  min_signal_rate = 0.02
  max_signal_rate = 0.95

  # architecture
  embedding_dims = 32
  embedding_max_frequency = 1000.0
  embedding_min_frequency = 1.0

  # optimization
  batch_size = 64
  ema = 0.999
  learning_rate = 1e-3
  weight_decay = 1e-4

  # dictionary related
  DICTIONARY_SIZE = 246
  TOKENS_CAPACITY = 2048

  c_dir = "./data/JL/"
  parsed_dir = "./data/parsed/"

  dataset_with_filenames = load_dataset(parsed_dir)
  dataset, filenames = dataset_with_filenames

  dataset[0] = scale(dataset[0], DICTIONARY_SIZE)

  network = get_network(TOKENS_CAPACITY, embedding_min_frequency, embedding_max_frequency, embedding_dims)

  model = DiffusionModel(TOKENS_CAPACITY, DICTIONARY_SIZE, network, batch_size, max_signal_rate, min_signal_rate, ema)
  model.compile(
      optimizer=keras.optimizers.experimental.AdamW(
          learning_rate=learning_rate, weight_decay=weight_decay
      ),
      loss=keras.losses.mean_absolute_error, # it might be worth to change loss given here
  )
  checkpoint_path = "checkpoints/diffusion_model"
  checkpoint_callback = keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_path,
      save_weights_only=True,
      monitor="val_kid",
      mode="min",
      save_best_only=True,
  )

  # calculate mean and variance of training dataset for normalization
  # TODO: get to know how to use this kind of normalizer and use it or discard it
  # model.normalizer.adapt(dataset)

  # run training and plot generated images periodically
  print(f"Dataset shape: {dataset.shape}")
  model.fit(
      dataset,
      batch_size=batch_size,
      epochs=num_epochs,
      callbacks=[
          checkpoint_callback,
      ],
  )
  

if __name__ == '__main__':
  main()