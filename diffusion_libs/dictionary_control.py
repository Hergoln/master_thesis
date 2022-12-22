NUM_LIT = "NUM"
STR_LIT = "STR"
EMPTY = "EMPTY"
vocabulary =[]
keywords = [
  "return", "auto", "break", "case", 
  "char", "const", "continue", "default", 
  "do", "double", "else", "enum", 
  "extern", "float", "for", "goto", "if", 
  "int", "long", "register", "return", 
  "short", "signed", "sizeof", "static", 
  "struct", "switch", "typedef", "union", 
  "unsigned", "void", "volatile", "while", 
  "true", "false", "printf", "scanf", "~", 
  "!", "#", "$", "%", "^", "&", "*", "(",
  ")", "_", "+", ",", ".", "/", "|","\\",
  "`", "-", "=", "<", ">", "?", "{", "}",
  "'", "[", "]", ":", ";"]

def loop_fill(type, capacity):
  for i in range(capacity):
    vocabulary.append(type + str(i))

def fill_vocabulary():
  vocabulary.append(EMPTY)
  vocabulary.append(NUM_LIT)
  vocabulary.append(STR_LIT)

  for keyword in keywords:
    vocabulary.append(keyword)

  loop_fill("ID", 128);
  loop_fill("TP", 32);
  loop_fill("RF", 16);

def val_to_word(val):
  return vocabulary[int(val)]

def convert_back_to_code(vector):
  return [val_to_word(v) for v in vector]