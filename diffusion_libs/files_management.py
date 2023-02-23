import numpy as np
import os
from multiprocessing.pool import ThreadPool

def prepare_record(parsed_file):
  # print(parsed_file)
  parsed = np.loadtxt(parsed_file, dtype=float, converters=float)
  return parsed

def load_dataset(parsed_dir, max_size=None, use_threads=True) -> list:
  parsed_files = sorted(os.listdir(parsed_dir))
  if max_size:
    parsed_files = parsed_files[:max_size]

  with ThreadPool() as pool:
    parsed_files = pool.map(lambda f: f"{parsed_dir}{f}", parsed_files)
    files = list(parsed_files)
  
  if use_threads:
    with ThreadPool() as pool:
    #   # pool.map guaranteese to preserve order
    #   # pool.map 'consumes' mapping created in previous with block
    #   # map() function returns a generator that is exhausted after is it used
      return [np.array(pool.map(lambda file: prepare_record(file), files)), files]
  else:
    l2 = []
    for file in files:
      l2.append(prepare_record(file))
  return [np.array(l2), files]

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