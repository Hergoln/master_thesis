def sql_simple_decode_sample(sql_s_vocabulary, sample):
  return [sql_s_vocabulary[int(v + 0.5)] for v in sample]


def sql_simple_decode_sample_into_text(sql_s_vocabulary, sample):
  return " ".join([sql_s_vocabulary[int(v + 0.5)] for v in sample])