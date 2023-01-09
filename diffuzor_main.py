from diffusion_libs import *
import tensorflow as tf
from tensorflow import keras

import argparse

import logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def parse():
    parser = argparse.ArgumentParser(description='Master thesis')
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
        num_epochs = 4096
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

        preprocessed_dataset = rescale(dataset, DICTIONARY_SIZE)
        print("Dataset preprocessed")

        model.compile(
            optimizer = keras.optimizers.experimental.AdamW(
                learning_rate=learning_rate, weight_decay=weight_decay
            ),
            # mse loss is pretty good for my model because it represents 
            # how close is prediction of my model to original sample
            # it represents "closeness of predicted code to original"
            # loss function cannot be simply MAE because most of samples
            # are EMPTY, this means full of EMPTY symbols thus anything
            # that we get will be compared to mostly EMPTY vector. I think I
            # have to use maybe attention mechanism or something that addresses
            # this issue
            loss = keras.losses.mean_absolute_error
        )
        print("Model compiled")
        
        checkpoint_path = "checkpoints\diffusion_model\cp-{epoch:04d}\model"
        
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_weights_only=True,
            monitor="i_loss",
            mode="min",
            save_best_only=False,
        )

        print("Started training")
        model.fit(
            preprocessed_dataset,
            batch_size=batch_size,
            epochs=num_epochs,
            callbacks=[
                checkpoint_callback,
                keras.callbacks.CSVLogger(f"checkpoints\\diffusion_model\\history.csv"),
                CustomCallback(checkpoint_path, 1, 20)
            ],
        )
        print("Completed training")
    except Exception as e:
        logger.exception(e)

if __name__ == '__main__':
  main()