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

int handleName(std::string);
std::vector<int> parse_initialization(const std::string);
std::vector<int> parse_reference(const std::string);
void fill_vocabulary();
int from_dict(std::string);
std::string decode(std::string);