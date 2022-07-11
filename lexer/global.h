#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

extern std::string NUM_LIT;
extern std::string STR_LIT;

extern std::stringstream ss;

extern int length;

struct Symbol {
  std::string name;
  std::string prefix;
  long no;
};

std::string handleName(std::string);
std::string parse_initialization(const std::string);
std::string parse_reference(const std::string);
void fill_vocabulary();
std::string from_dict(std::string);
std::string decode(std::string);