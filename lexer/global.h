#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

extern std::string NUM_LIT;
extern std::string STR_LIT;

extern std::stringstream ss;

extern int length;
  
const std::string DEFAULT_TOO_LONG_FILE = "too_long_functions.log";
const int DEFAULT_VECTOR_LENGTH = 2048; // Till i do not get more samples from doctor I can't say if it is small or big number
const std::string EMPTY_TOKEN = "0";

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
std::string find_arg(char**, int, const std::string&);
std::string find_single_arg(char**, int, const std::string&);
