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

extern std::string handleName(std::string);
std::string parse_initialization(const std::string yytext, char delimiter);