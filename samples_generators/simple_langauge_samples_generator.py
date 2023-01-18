import numpy as np
import random

ops = ['-', '+', '/', '*', '^']
EMPTY = 'EMPTY'
NUM = 'NUM'
EQ_SIGN = '='
sl_vocabulary = ops + [EQ_SIGN, NUM, EMPTY] # has to be in this order, DO NOT TOUCH
dictionary = {el:idx for idx,el in enumerate(sl_vocabulary)}

def sl_gen_sample(length, max_vector_len):
  values = []
  for _ in range(length):
    values.append(dictionary[NUM])
    values.append(random.randint(0, len(ops)-1))
  values += [dictionary[NUM],dictionary[EQ_SIGN],dictionary[NUM]]
  filled_to_the_brim = values + [dictionary[EMPTY] for _ in range(max_vector_len - len(values))]
  return filled_to_the_brim

def sl_generate_samples(num_samples, min_len, max_len, max_vector_len):
  lens = np.random.randint(min_len, max_len, num_samples)
  return np.asarray([sl_gen_sample(l, max_vector_len) for l in lens])

def sl_decode_sample(sample):
  return [sl_vocabulary[int(v + 0.5)] for v in sample]

def sl_decode_sample_into_text(sample):
  return " ".join([sl_vocabulary[v] for v in sample])