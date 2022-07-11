#include "global.h"
#include <algorithm>
#include <vector>

int length;

std::vector<Symbol> symtable;

std::vector<std::string> vocabulary;
std::vector<std::string> keywords {
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
  "[", "]", ":", ";"};

std::string NUM_LIT = "NUM";
std::string STR_LIT = "STR";

void loop_fill(std::string type, int capacity) {
  for(int i = 0; i < capacity; ++i)
    vocabulary.push_back(type + std::to_string(i));
}

void fill_vocabulary() {
  vocabulary.push_back(NUM_LIT);
  vocabulary.push_back(STR_LIT);

  for(auto &keyword : keywords)
    vocabulary.push_back(keyword);

  loop_fill("ID", 128);
  loop_fill("TP", 32);
  loop_fill("RF", 16);
}


std::string decode(std::string id) {
  return vocabulary[std::stoi(id)];
}

std::string from_dict(std::string element) {
  if(std::find(vocabulary.begin(), vocabulary.end(), element) - vocabulary.begin() == 244)
    std::cout << "el : " << element << std::endl;

  length++;
  return std::to_string(std::find(vocabulary.begin(), vocabulary.end(), element) - vocabulary.begin()) + " ";
}

std::string handle_name(const std::string yytext, const std::string prefix) {
  std::string element = prefix;
  if (std::find(keywords.begin(), keywords.end(), yytext) != keywords.end()) {
    return from_dict(yytext);
  }
  auto it = std::find_if(symtable.begin(), symtable.end(), [&yytext](const Symbol sym){ return yytext.compare(sym.name) == 0; });

  if(it == symtable.end()) {
    long no = symtable.size();
    symtable.push_back({yytext, prefix, no});
    element += std::to_string(no);
  } else {
    element += std::to_string((*it).no);
  }

  return from_dict(element);
}

std::string parse_initialization(const std::string yytext) {
  char del = ' ';
  std::vector<std::string> parsed;
  std::stringstream ss(yytext);
  std::string item, result;

  while(getline(ss, item, del)) parsed.push_back(item);
  parsed[0] = handle_name(parsed[0], "TP");
  parsed[1] = handle_name(parsed[1], "ID");

  return parsed[0] + parsed[1];
}

std::string parse_reference(const std::string yytext) {
  char del = '.';
  std::vector<std::string> parsed;
  std::stringstream ss(yytext);
  std::string item, result;

  while(getline(ss, item, del)) parsed.push_back(item);

  result = handle_name(parsed[0], "ID");
  for (int i = 1; i < parsed.size(); ++i) {
    result += from_dict(".RF" + std::to_string(i - 1));
  }
  
  return result;
} 