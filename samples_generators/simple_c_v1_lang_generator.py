isFilled = False
vocabulary_c_v1 = [
  "void", "int", "char", "STRING", "NUM", "return", "printf", "scanf", 
  "-", "+", "/", "*", "&", "=", "(", ")", ";", "{", "}", "EMPTY", ","
]


def fill_vocabulary_c_v1():
  global isFilled
  if isFilled:
    print("Dictionary already filled")
    return


  for index in range(16):
    current_id = "ID" + str(index)
    vocabulary_c_v1.append(current_id)

  # will add functions in some further generation
  # for index in range(8):
  #   current_fun = "FUN" + str(index)
  #   vocabulary.append(current_fun)
  #   funs.append(current_fun)
  isFilled = True

def convert_back_to_code_c_v1(vector):
  return [vocabulary_c_v1[int(v + 0.5)] for v in vector]