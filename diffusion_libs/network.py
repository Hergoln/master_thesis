import math
import tensorflow as tf
from tensorflow import keras
from keras import layers

def sinusoidal_embedding(x, embedding_min_frequency, embedding_max_frequency, embedding_dims):
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=1
    )
    return embeddings
  
# def get_network(tokens_capacity, embedding_min_frequency, embedding_max_frequency, embedding_dims):
#     noisy_images = keras.Input(shape=(tokens_capacity))
#     noise_variances = keras.Input(shape=(1))

#     emb = lambda x: sinusoidal_embedding(x, embedding_min_frequency, embedding_max_frequency, embedding_dims)
#     e = layers.Lambda(emb)(noise_variances)

#     x = layers.Dense(1024)(noisy_images)
#     x = layers.Concatenate()([x, e])
#     x = layers.Dense(512, name="dense01")(x)
#     x = layers.Dense(1024, name="dense02", activation=keras.activations.relu)(x)
#     x = layers.Dense(2048, name="dense03", activation=keras.activations.relu)(x)
#     x = layers.Dense(2048, name="last_dense")(x)

#     return keras.Model([noisy_images, noise_variances], x, name="simple_net")


def get_network(tokens_capacity, widths, block_depth, embedding_min_frequency, embedding_max_frequency, embedding_dims):
    noisy_images = keras.Input(shape=(tokens_capacity))
    noise_variances = keras.Input(shape=(1))

    emb = lambda x: sinusoidal_embedding(x, embedding_min_frequency, embedding_max_frequency, embedding_dims)
    e = layers.Lambda(emb)(noise_variances)

    x = layers.Dense(widths[0])(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Dense(2048, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")

def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[0]
        if input_width == width:
            residual = x
        else:
            residual = layers.Dense(width)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Dense(
            width, activation=keras.activations.swish
        )(x)
        x = layers.Dense(width)(x)
        x = layers.Add()([x, residual])
        return x

    return apply

def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.Dense(width // 2)(x)
        return x

    return apply

def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.Dense(width * 2)(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply