import numpy as np
import os
from multiprocessing.pool import ThreadPool

def prepare_record(parsed_file):
  parsed = np.loadtxt(parsed_file, dtype=float, converters=float)
  return parsed

def load_dataset(parsed_dir) -> list:
  parsed_files = sorted(os.listdir(parsed_dir))

  with ThreadPool() as pool:
    parsed_files = pool.map(lambda f: f"{parsed_dir}{f}", parsed_files)
    files = list(parsed_files)

  with ThreadPool() as pool:
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