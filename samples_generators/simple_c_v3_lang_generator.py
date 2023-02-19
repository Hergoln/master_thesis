isFilled3 = False
vocabulary_c_v3 = [
  "EMPTY", "void", "int", "char", "STRING", "NUM",
  "-", "+", "/", "*", "&", "|", "=", "(", ")", ";", "{", "}", ",", 
  "!", "<", ">", "return", "printf", "scanf", "if", "else", "while",
  "for"
]


def fill_vocabulary_c_v3():
  global isFilled3
  if isFilled3:
    print("Dictionary already filled")
    return

  for index in range(16):
    current_id = "ID" + str(index)
    vocabulary_c_v3.append(current_id)

  isFilled3 = True

def convert_back_to_code_c_v3(vector):
  return [vocabulary_c_v3[int(v + 0.5)] for v in vector]