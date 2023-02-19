isFilled4 = False
vocabulary_c_v4 = [
  "EMPTY", "void", "int", "char", "STRING", "NUM",
  "-", "+", "/", "*", "&", "|", "=", "(", ")", ";", "{", "}", ",", 
  "!", "<", ">", "return", "printf", "scanf", "if", "else", "while",
  "for", "[", "]"
]


def fill_vocabulary_c_v4():
  global isFilled4
  if isFilled4:
    print("Dictionary already filled")
    return

  for index in range(16):
    current_id = "ID" + str(index)
    vocabulary_c_v4.append(current_id)

  isFilled4 = True

def convert_back_to_code_c_v4(vector):
  return [vocabulary_c_v4[int(v + 0.5)] for v in vector]