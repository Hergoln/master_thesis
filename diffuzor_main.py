from diffusion_libs import *
from tensorflow import keras

import logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

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

    try:
        print("Started loading dataset")
        dataset_with_filenames = load_dataset(parsed_dir)
        print("Loaded dataset")
        dataset, filenames = dataset_with_filenames

        dataset[0] = scale(dataset[0], DICTIONARY_SIZE)
        print(f"Dataset shape: {dataset.shape}")

        network = get_network(TOKENS_CAPACITY, embedding_min_frequency, embedding_max_frequency, embedding_dims)
        print("Network created")

        model = DiffusionModel(TOKENS_CAPACITY, DICTIONARY_SIZE, network, batch_size, max_signal_rate, min_signal_rate, ema)
        print("Model created")

        model.compile(
            optimizer=keras.optimizers.experimental.AdamW(
                learning_rate=learning_rate, weight_decay=weight_decay
            ),
            loss=keras.losses.mean_absolute_error, # it might be worth to change loss given here
        )
        print("Model compiled")

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

        # run training
        model.fit(
            dataset,
            batch_size=batch_size,
            epochs=num_epochs,
            callbacks=[
                checkpoint_callback,
            ],
        )
        print("Completed training")
    except Exception as e:
        logger.exception(e)

if __name__ == '__main__':
  main()