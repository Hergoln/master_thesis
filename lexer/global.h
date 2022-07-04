#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <vector>

struct Symbol {
  std::string name;
  std::string prefix;
  long no;
};

std::vector<Symbol> symtable;
std::stringstream ss;

std::vector<std::string> keywords {
  "return", "auto", "break", "case", 
  "char", "const", "continue", "default", 
  "do", "double", "else", "enum", 
  "extern", "float", "for", "goto", "if", 
  "int", "long", "register", "return", 
  "short", "signed", "sizeof", "static", 
  "struct", "switch", "typedef", "union", 
  "unsigned", "void", "volatile", "while", 
  "true", "false", "printf", "scanf"};

std::string handleName(std::string);
std::string parse_initialization(const std::string yytext, char delimiter);
std::string parse_reference(const std::string yytext);