import numpy as np
import os
from multiprocessing.pool import ThreadPool
import itertools

def prepare_record(parsed_file):
  parsed = np.loadtxt(parsed_file, dtype=float, converters=float)
  return parsed


def load_dataset(parsed_dir) -> list:
  parsed_files = sorted(os.listdir(parsed_dir))
  # files_batches = [parsed_files[i: i + MAX_FILES_PER_LOAD] for i in range(0, len(parsed_files), MAX_FILES_PER_LOAD)]
  # parsed_files = parsed_files[MAX_FILES_PER_LOAD + 100: MAX_FILES_PER_LOAD*2]

  with ThreadPool() as pool:
    parsed_files = pool.map(lambda f: f"{parsed_dir}{f}", parsed_files)
    files = list(parsed_files)

  with ThreadPool() as pool:
  #   # pool.map guaranteese to preserve order
  #   # pool.map 'consumes' mapping created in previous with block
  #   # map() function returns a generator that is exhausted after is it used
    return [np.array(pool.map(lambda file: prepare_record(file), files)), files]

def split_files_functions(input_f):
  cntr = 0
  started = False
  f_cntr = 0
  prev = 0

  with open(input_f, 'r') as f:
    input = f.read()
    for c in range(len(input)):
      if input[c] == '{':
        started = True
        cntr += 1
      if input[c] == '}':
        cntr -= 1
        if cntr == 0 and started:
          with open(f'{input_f}{f_cntr}.f', 'w') as f2:
            f2.write(input[prev:c + 1])
          prev = c + 1
          f_cntr += 1