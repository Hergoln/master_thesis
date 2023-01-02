from diffusion_libs import *
from tensorflow import keras

import argparse

import logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def parse():
    parser = argparse.ArgumentParser(description='Python CardGame for zkum classes')
    parser.add_argument('--dev', action='store_true', help='Development mode. Only a fraction of dataset is loaded and number of epochs is minimized')
    return parser.parse_args()

def main():
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

    args = parse()
    if args.dev:
        num_epochs = 4
        max_size = batch_size
        dataset_trimmer = lambda dataset, batch_size: dataset[:batch_size]
    else:
        num_epochs = 128
        max_size = None
        dataset_trimmer = lambda dataset, batch_size: dataset[:batch_size * (len(dataset)//batch_size)]

    try:
        print("Started loading dataset")
        dataset, filenames = load_dataset(parsed_dir, max_size)
        # dataset size has to be a multiplication of batch_size
        dataset = dataset_trimmer(dataset, batch_size)
        print("Loaded dataset")
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

        checkpoint_path = " "
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor="loss", # probably should change it 
            mode="min", # this should go with change of monitor value
            save_best_only=True,
        )

        print("Started training")
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