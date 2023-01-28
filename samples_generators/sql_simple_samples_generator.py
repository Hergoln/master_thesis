import numpy as np
import random

cols = ['COL_ZERO', 'COL_TWO', 'COL_THREE', 'COL_FOUR', 'COL_FIVE', 'COL_SIX', 'COL_SEVEN']
frm = 'FROM'

literals = ['STRING', 'NUM']
concats = ['AND', 'OR']

sql_s_vocabulary = ['SELECT'] + cols + [',',
    'FROM',
    'TABLE_NAME',
    'WHERE',
    'IN',
    'EXISTS',
    # 'JOIN', # joins introduce a lot of complexity, will try without it and if it goes well add it later
    'AND',
    'OR',
    'LIKE',
    '(',
    'STRING',
    'NUM',
    ')',
    'LIMIT',
    'EMPTY'
  ]

sqlSDict = {el:idx for idx,el in enumerate(sql_s_vocabulary)}

global_recursive_counter = 0

def select_col_not_in(in_cols):
  cc = [it for it in cols if it not in in_cols]
  return random.choice(cc)

def gen_core():
  q = []
  q.append('SELECT')
  c = cols[random.randint(0, len(cols)-1)]
  output_cols = [c]
  q.append(c)
  temp_cols = [it for it in cols if it not in output_cols]
  for cc in random.sample(temp_cols, k=random.randint(0, len(temp_cols)-1)):
    q.append(',')
    q.append(cc)
    output_cols.append(cc)
  q.append(frm)
  q.append('TABLE_NAME')
  return q, output_cols

def gen_IN_conditions(in_cols):
  if random.random() < 0.5:
    val_type = 'STRING'
  else:
    val_type = 'NUM'
  q = []
  q.append(random.choice(in_cols))
  q.append('IN')
  q.append('(')
  q.append(val_type)
  for _ in range(random.randint(0, 16)):
    q.append(',')
    q.append(val_type)
  q.append(')')
  return q


def gen_EXISTS_conditions(in_cols):
  global global_recursive_counter
  global_recursive_counter += 1
  if global_recursive_counter > 8:
    return gen_LIKE_conditions(in_cols)
  q = []
  q.append('EXISTS')
  q.append('(')
  q.append('SELECT')
  q.append(random.choice(in_cols))
  q.append(frm)
  q.append('TABLE_NAME')
  q.append('WHERE')
  q += gen_condition(in_cols)
  q.append(')')
  global_recursive_counter = 0
  return q


def gen_LIKE_conditions(in_cols):
  q = []
  q.append(random.choice(in_cols))
  q.append('LIKE')
  q.append('STRING')
  return q


def gen_condition(in_cols):
  possible_conditions = [[0.1, gen_IN_conditions], [0.11, gen_LIKE_conditions], [1.0, gen_EXISTS_conditions]]
  rnJesus = random.random()
  for condition in possible_conditions:
    if rnJesus < condition[0]:
      return condition[1](in_cols)


def gen_conditions(in_cols):
  q = []

  q.append('WHERE')
  q += gen_condition(in_cols)
  return q


def sql_simple_gen_sample(max_len_of_samples=None):
  q, used_cols = gen_core()
  q += gen_conditions(used_cols)
  query = [sqlSDict[it] for it in q]
  not_filled_query = [sqlSDict[it] for it in q]
  if max_len_of_samples is not None:
    empty_values = [sqlSDict['EMPTY'] for _ in range(max_len_of_samples - len(query))]
    query += empty_values

  return query, not_filled_query


def sql_simple_generate_samples(num_of_samples, max_len_of_samples):
  return [sql_simple_gen_sample(max_len_of_samples) for _ in range(num_of_samples)]


def check_max_length():
  lens = [len(sql_simple_gen_sample()) for _ in range(1000)]
  return max(lens)

def sql_simple_decode_sample(sample):
  return [sql_s_vocabulary[int(v + 0.5)] for v in sample]


def sql_simple_decode_sample_into_text(sample):
  return " ".join([sql_s_vocabulary[int(v + 0.5)] for v in sample])