from diffusion_libs import *
from samples_generators import convert_back_to_code_c_v1, fill_vocabulary_c_v1, remove_token_and_shift_sample_randomized, ErrorsIntroducer, vocabulary_c_v1
from samples_generators import convert_back_to_code_c_v2, fill_vocabulary_c_v2, vocabulary_c_v2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow as tf

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
    parser.add_argument('--errors_learning', action='store_true', help='Is learning errors fix time. If yes, error introducer will be created and model will learn how to fix bugs')
    return parser.parse_args()


def main():
    fill_vocabulary()
    fill_vocabulary_c_v1()
    fill_vocabulary_c_v2()
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
    learning_rate = 1e-3

    # dictionary related
    DICTIONARY_SIZE = 43
    TOKENS_CAPACITY = 256

    widths = [64, 64, 96, 128]
    block_depth = 2

    data_dir = f"./data/simple_c_v2/"
    lang_base = f"checkpoints/simple_c_v2"

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
        # network.summary()

        dictionary = {el:idx for idx,el in enumerate(vocabulary_c_v2)}
        use_xy = False
        dataset = scale_dataset_down(dataset, DICTIONARY_SIZE)
        print(f"dataset min: {tf.reduce_min(dataset)}")
        print(f"dataset max: {tf.reduce_max(dataset)}")
        dataset_for_normalization = dataset
        if args.errors_learning:
            remove_tokens_introducer = remove_token_and_shift_sample_randomized([";", "+", "-", "/", "="], 0.5, dictionary, TOKENS_CAPACITY)
            errorIntroducer = ErrorsIntroducer([remove_tokens_introducer])
            
            bugged_dataset = errorIntroducer.apply(dataset)
            dataset = np.asarray([bugged_dataset, dataset])
            lang_base += "_fix_bug"
            use_xy = True

        model = DiffusionModel(
                TOKENS_CAPACITY, DICTIONARY_SIZE, network, batch_size, max_signal_rate, 
                min_signal_rate, ema, use_xy
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
            checkpoint_base_path, 5, 100, converter=convert_back_to_code_c_v2, scaler=scaler_up,
            history_path=f"{lang_base}/history.csv", append_history=is_loading
        )

        print(f"Normalized dataset shape: {dataset_for_normalization.shape}")
        model.normalizer = resolve_normalization(
            args.compute_normalizer, TOKENS_CAPACITY,
            file=f"{lang_base}/normalizer_weights.npy", dataset=dataset_for_normalization
        )

        if is_loading:
            print("loaded weights")
            model.load_weights(args.load)

        print("saving parameters")
        with open(f"{lang_base}/params.txt", 'w') as fHandler:
            fHandler.write("min_signal_rate = " + str(min_signal_rate) + "\n")
            fHandler.write("max_signal_rate = " + str(max_signal_rate) + "\n")
            fHandler.write("embedding_dims = " + str(embedding_dims) + "\n")
            fHandler.write("embedding_max_frequency = " + str(embedding_max_frequency) + "\n")
            fHandler.write("embedding_min_frequency = " + str(embedding_min_frequency) + "\n")
            fHandler.write("batch_size = " + str(batch_size) + "\n")
            fHandler.write("ema = " + str(ema) + "\n")
            fHandler.write("learning_rate = " + str(learning_rate) + "\n")
            fHandler.write("DICTIONARY_SIZE = " + str(DICTIONARY_SIZE) + "\n")
            fHandler.write("TOKENS_CAPACITY = " + str(TOKENS_CAPACITY) + "\n")
            fHandler.write("widths = " + str(widths) + "\n")
            fHandler.write("block_depth = " + str(block_depth) + "\n")

        print("Started training")
        if use_xy:
            model.fit(
            dataset[0],
            dataset[1],
            batch_size=batch_size,
            epochs=num_epochs,
            callbacks=[
                checkpoint_callback,
                sample_generator_callback
            ],
        )
        else:
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