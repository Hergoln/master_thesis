from diffusion_libs import *
import tensorflow as tf
from tensorflow import keras
import numpy as np

import argparse

import logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def resolve_normalization(compute, tokens_capacity, file, dataset):
    if compute:
        layer = keras.layers.Normalization()
        layer.adapt(dataset)
        w = layer.get_weights()
        w = np.asarray(w[:-1])
        np.save(file, w)
        print("adapted normalizer") 
    else:
        n_w = np.load(file, allow_pickle=True)
        print(n_w.shape)
        layer = keras.layers.Normalization(mean=n_w[0], variance=n_w[1])
        print("loaded normalizer")
    layer.build((tokens_capacity))
    return layer


def parse():
    parser = argparse.ArgumentParser(description='Master thesis')
    parser.add_argument('--dev', action='store_true', help='Development mode. Only a fraction of dataset is loaded and number of epochs is minimized')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--compute_normalizer', action='store_true', help='Compute normalizer. If not set than will look for weight of normalizer.')
    parser.add_argument('--load', type=str, help='path to model')
    return parser.parse_args()


def main():
    fill_vocabulary()
    # sampling
    min_signal_rate = 0.02
    max_signal_rate = 0.95

    # architecture
    embedding_dims = 32
    embedding_max_frequency = 1000.0
    embedding_min_frequency = 1.0

    # optimization
    batch_size = 16
    ema = 0.999
    learning_rate = 1e-5

    # dictionary related
    DICTIONARY_SIZE = 246 # only issue is that it displays different value because of floats precision
    TOKENS_CAPACITY = 2048

    widths = [8, 16, 32, 128, 256, 1024]
    block_depth = 2

    data_dir = f"./data/parsed/"
    lang_base = f"checkpoints/c_lang"

    args = parse()
    if args.dev:
        num_epochs = 1
        max_size = batch_size
        dataset_trimmer = lambda dataset, batch_size: dataset[:batch_size]
    else:
        num_epochs = 4096 if args.epochs is None else args.epochs
        max_size = None
        dataset_trimmer = lambda dataset, batch_size: dataset[:batch_size * (len(dataset)//batch_size)]

    if args.load:
        is_loading = True
    else:
        is_loading = False

    try:
        print("Started loading dataset")
        dataset, filenames = load_dataset(data_dir, max_size)
        # dataset size has to be a multiplication of batch_size
        dataset = dataset_trimmer(dataset, batch_size)
        print("Loaded dataset")
        print(f"Dataset shape: {dataset.shape}")

        network = get_network(
                TOKENS_CAPACITY, embedding_min_frequency, embedding_max_frequency, 
                embedding_dims, widths=widths, block_depth=block_depth, name="complicated"
            )
        print("Network created")
        network.summary()

        model = DiffusionModel(
                TOKENS_CAPACITY, DICTIONARY_SIZE, network, batch_size, max_signal_rate, 
                min_signal_rate, ema
            )
        print("Model created")

        model.compile(
            optimizer = keras.optimizers.experimental.Adam(
                learning_rate=learning_rate
            ),
            loss = keras.losses.mean_absolute_error
        )
        print("Model compiled")
        checkpoint_base_path = str(lang_base) + "/cp-{epoch:04d}"
        checkpoint_path = str(lang_base) + "/cp-{epoch:04d}/model"
        
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_weights_only=True,
            monitor="i_loss",
            mode="min",
            save_best_only=False,
        )

        scaler_up = lambda x: scale_dataset(x, DICTIONARY_SIZE)
        sample_generator_callback = SaveSamplesCallback(
            checkpoint_base_path, 5, 100, converter=convert_back_to_code, scaler=scaler_up,
            history_path=f"{lang_base}/history.csv", append_history=is_loading
        )

        dataset = scale_dataset_down(dataset, DICTIONARY_SIZE)
        print(f"dataset min: {tf.reduce_min(dataset)}")
        print(f"dataset max: {tf.reduce_max(dataset)}")

        model.normalizer = resolve_normalization(
            args.compute_normalizer, TOKENS_CAPACITY,
            file=f"{lang_base}/normalizer_weights.npy", dataset=dataset
        )

        if is_loading:
            print("loaded weights")
            model.load_weights(args.load)

        print("Started training")
        model.fit(
            dataset,
            batch_size=batch_size,
            epochs=num_epochs,
            callbacks=[
                checkpoint_callback,
                sample_generator_callback
            ],
        )
        print("Completed training")
    except Exception as e:
        logger.exception(e)

if __name__ == '__main__':
  main()