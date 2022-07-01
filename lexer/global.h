#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <vector>

struct Symbol {
  std::string name;
  long no;
};

std::vector<Symbol> symtable;

extern std::string handleName(std::string);