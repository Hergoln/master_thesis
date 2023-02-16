isFilled = False
vocabulary_c_v2 = [
  "void", "int", "char", "STRING", "NUM", "return", "printf", "scanf", 
  "-", "+", "/", "*", "&", "|", "=", "(", ")", ";", "{", "}", ",", 
  "if", "else", "!", "<", ">", "EMPTY"
]


def fill_vocabulary_c_v2():
  global isFilled
  if isFilled:
    print("Dictionary already filled")
    return

  for index in range(16):
    current_id = "ID" + str(index)
    vocabulary_c_v2.append(current_id)

  isFilled = True

def convert_back_to_code_c_v2(vector):
  return [vocabulary_c_v2[int(v + 0.5)] for v in vector]