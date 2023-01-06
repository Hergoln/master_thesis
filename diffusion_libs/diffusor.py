import tensorflow as tf
from tensorflow import keras
import numpy as np

def isScaled(dataframe):
    for val in dataframe:
        if val < 0 or val > 1:
            return False
    return True

def rescale(dataframe, dic_size):
  return (dataframe * 2 / (dic_size - 1)) - 1

def scale(dataframe, dic_size):
  return (dataframe + 1) * (dic_size - 1) / 2

def scale_dataset(dataframe, dic_size):
    l = lambda x: (x + 1) * (dic_size - 1) / 2
    return np.array(list(map(l, dataframe)))

class DiffusionModel(keras.Model):
    def __init__(
      self, tokens_capacity, dictionary_size, network, batch_size, 
      max_signal_rate, min_signal_rate, ema, 
    ):
        super().__init__()

        self.tokens_capacity = tokens_capacity
        self.dictionary_size = dictionary_size
        self.network = network
        self.batch_size = batch_size
        self.max_signal_rate = max_signal_rate
        self.min_signal_rate = min_signal_rate
        self.ema = ema
        self.ema_network = keras.models.clone_model(self.network)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss") # for training
        self.sample_loss_tracker = keras.metrics.Mean(name="i_loss") # for human evaluation

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.sample_loss_tracker]

    def normalize(self, samples):
        return tf.math.divide(samples * 2, (self.dictionary_size - 1)) - 1

    def denormalize(self, samples):
        vals = tf.math.divide((samples + 1) * (self.dictionary_size - 1), 2)

        return tf.clip_by_value(vals, 0.0, (self.dictionary_size - 1))

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(self.max_signal_rate)
        end_angle = tf.acos(self.min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_samples, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the sample component using it
        pred_noises = network([noisy_samples, noise_rates**2], training=training)
        pred_samples = (noisy_samples - noise_rates * pred_noises) / signal_rates # maybe some more sophisticated way of removing noise

        return pred_noises, pred_samples

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_samples = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy sample" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_samples = initial_noise
        for step in range(diffusion_steps):
            noisy_samples = next_noisy_samples

            # separate the current noisy sample to its components
            diffusion_times = tf.ones((num_samples)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_samples = self.denoise(
                noisy_samples, noise_rates, signal_rates, training=False
            )
            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_samples = (
                next_signal_rates * pred_samples + next_noise_rates * pred_noises
            )
            # this new noisy sample will be used in the next step

        return pred_samples

    def generate(self, num_samples, diffusion_steps):
        # Generate sample from complete noise
        initial_noise = tf.random.normal(shape=(num_samples, self.tokens_capacity))
        generated_sample = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_sample = self.denormalize(generated_sample)
        return generated_sample

    def train_step(self, samples):
        # normalize samples to have standard deviation of 1, like the noises
        # TODO: my normalization does not create standard deviation value range though
        samples = self.normalize(samples)
        noises = tf.random.normal(shape=(self.batch_size, self.tokens_capacity))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the samples with noises accordingly
        noisy_samples = signal_rates * samples + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy samples to their components
            pred_noises, pred_samples = self.denoise(
                noisy_samples, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            sample_loss = self.loss(samples, pred_samples)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.sample_loss_tracker.update_state(sample_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, samples):
        samples = self.normalize(samples)
        noises = tf.random.normal(shape=(self.batch_size, self.tokens_capacity))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the samples with noises accordingly
        noisy_samples = signal_rates * samples + noise_rates * noises

        # use the network to separate noisy samples to their components
        pred_noises, pred_samples = self.denoise(
            noisy_samples, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        sample_loss = self.loss(samples, pred_samples)

        self.sample_loss_tracker.update_state(sample_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        return {m.name: m.result() for m in self.metrics}