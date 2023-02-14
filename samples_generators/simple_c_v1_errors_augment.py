import numpy as np
import random

def remove_token_and_shift_sample_randomized(tokens_to_remove, chance_to_survive, dictionary, sample_len):
  EMPTY = dictionary["EMPTY"]
  tokens_to_remove = [dictionary[token] for token in tokens_to_remove]
  def apply(samples):
    to_return = [[token for token in sample if (token not in tokens_to_remove or random.random() < chance_to_survive)] for sample in samples]
    to_return = [sample + [EMPTY for _ in range(sample_len - len(sample))] for sample in to_return]
    return to_return
  return apply

class ErrorsIntroducer:
  def __init__(self, errors_introducers) -> None:
    self.errors_introducers = errors_introducers

  '''
  no_of_bugged_samples - should be used only in case introducers are based on some kind of randomness and You want to have multiple takes on the same 
  '''
  def apply(self, samples):
    for introducer in self.errors_introducers:
      samples = introducer(samples)
    return np.asarray(samples)