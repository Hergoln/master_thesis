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
  
def get_network(tokens_capacity, embedding_min_frequency, embedding_max_frequency, embedding_dims):
    noisy_images = keras.Input(shape=(tokens_capacity))
    noise_variances = keras.Input(shape=(1))

    emb = lambda x: sinusoidal_embedding(x, embedding_min_frequency, embedding_max_frequency, embedding_dims)
    e = layers.Lambda(emb)(noise_variances)

    x = layers.Dense(64)(noisy_images)
    x = layers.Concatenate()([x, e])
    x = layers.Dense(1024, kernel_initializer="zeros")(x)
    x = layers.Dense(2048, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="simple_net")