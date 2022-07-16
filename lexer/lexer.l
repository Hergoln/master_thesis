%{
#include "global.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>

struct SAMPLE {
  std::string tokens;
  int length;
} SAMPLE;

std::stringstream ss;
std::vector<std::string> samples;

int DONE;
int brack_cnt = 0;
bool no_brack_yet = true;
extern int length; // this value is temporary and is used only to estimate average length of the vectorized representation
int vector_length;

const int DEFAULT_VECTOR_LENGTH = 512; // Till i do not get more samples from doctor I can't say if it is small or big number
const std::string EMPTY_TOKEN = "0";

std::string find_arg(char** argv, int argc, const std::string & option) {
    char ** itr = std::find(argv, argv+argc, option);
    if (itr != argv+argc && ++itr != argv+argc)
    {
        return std::string(*(itr));
    }
    return "";
}

void save_samples(std::vector<std::string> samples, std::string outfile, int vector_length) {
  std::ofstream ofs;
  for (int i =0; i < samples.size(); ++i) {
    std::string sample = samples[i];

    // int l = length;
    // while(l++ < vector_length) ss << EMPTY_TOKEN << " ";
    ofs.open(std::to_string(i) + outfile, std::ofstream::out);
    ofs << sample;
    ofs.close();
  }
}

%}

%option noyywrap

delim   [ \t]
D			  [0-9]
L			  [a-zA-Z_]
H			  [a-fA-F0-9]
E			  [Ee][+-]?{D}+
FS			(f|F|l|L)
IS			(u|U|l|L)*

includ  #include[ \t]*<.*>

comment \/\*([^*]|\*[^/])*\*\/|\/\/.*
string  \"([^\\"]|\\.)*\"

ID       {L}({L}|{D})*

%%
\r?\n                         {}
{includ}                      {}
{comment}                     {}
{delim}                       {}

{string}                      { ss << from_dict(STR_LIT);}

0[xX]{H}+{IS}?		            { ss << from_dict(NUM_LIT);} // numbers
0{D}+{IS}?		                { ss << from_dict(NUM_LIT);}  // numbers
{D}+{IS}?		                  { ss << from_dict(NUM_LIT);}  // numbers
{D}+{E}{FS}?	              	{ ss << from_dict(NUM_LIT);} // numbers
{D}*"."{D}+({E})?{FS}?	      { ss << from_dict(NUM_LIT);} // floating numbers
{D}+"."{D}*({E})?{FS}?	      { ss << from_dict(NUM_LIT);} // floating numbers

{ID}[ ]{ID}                   { ss << parse_initialization(yytext);}
{ID}("."{ID})*	              { ss << parse_reference(yytext);}
<<EOF>>                       { return DONE; }
.                             { 
  if (!strcmp("{", yytext)) {
    no_brack_yet = false;
    brack_cnt++;
  }
    
  if (!strcmp("}", yytext)) 
    brack_cnt--;
    
  ss << from_dict(yytext); // should be before any modifications because it will add '}' to stream

  if (brack_cnt == 0 && !no_brack_yet) {
    no_brack_yet = true;
    samples.push_back(ss.str());
    ss.str("");
  }
}
%%


int main(int argc, char* argv[]) {
  std::string outfile;
  std::string option = find_arg(argv, argc, "vector-length");
  if(!option.empty()) {
    vector_length = stoi(option);
    std::cout << vector_length << std::endl;
  } else {
    vector_length = DEFAULT_VECTOR_LENGTH;
  }

  option = find_arg(argv, argc, "outfile");
  if (option.empty()) {
    std::cerr << "Didn't receive output file" << std::endl;
    outfile = "test_out.parsed";
  } else {
    outfile += option;
  }

  fill_vocabulary();
  int res = yylex(); // lexing magic happens here


  save_samples(samples, outfile, vector_length);

  std::cout << "Write to file: " << outfile << std::endl;
  std::cout << "No. of tokens saved: " << length << std::endl;
  return res;
}