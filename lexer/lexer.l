%{
#include "global.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <memory>

typedef std::shared_ptr<std::vector<int>> SAMPLE;

SAMPLE crrs;
std::vector<SAMPLE> samples;

int DONE;
int brack_cnt = 0;
bool no_brack_yet = true;
int vector_length;

const std::string DEFAULT_TOO_LONG_FILE = "too_long_functions.log";
std::string CURRENT_FILE = "NOT_SPECIFIED";

const int DEFAULT_VECTOR_LENGTH = 512; // Till i do not get more samples from doctor I can't say if it is small or big number
const std::string EMPTY_TOKEN = "0";


void save_samples(std::vector<SAMPLE> samples, std::string outfile, int vector_length) {
  std::ofstream ofs;
  std::ofstream too_long_ofs;

  too_long_ofs.open(DEFAULT_TOO_LONG_FILE, std::ofstream::app);

  for (int i =0; i < samples.size(); ++i) {
    SAMPLE sample = samples[i];

    if (sample->size() > vector_length) {
      too_long_ofs << CURRENT_FILE << ": function no.: " << i << std::endl;
      continue;
    }

    ofs.open(std::to_string(i) + outfile, std::ofstream::out);
    // stream tokens into ofstream
    for (int t : *sample) {
      ofs << std::to_string(t) << " ";
    }

    // fill with EMPTY
    for(int i = 0; i < vector_length - sample->size(); ++i)
      ofs << EMPTY_TOKEN << " ";

    ofs.close();
  }
  too_long_ofs.close();
}

void psh_t(SAMPLE sample, int token) {
  sample->push_back(token);
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

{string}                      { psh_t(crrs, from_dict(STR_LIT));}

0[xX]{H}+{IS}?		            { psh_t(crrs, from_dict(NUM_LIT));} // numbers
0{D}+{IS}?		                { psh_t(crrs, from_dict(NUM_LIT));}  // numbers
{D}+{IS}?		                  { psh_t(crrs, from_dict(NUM_LIT));}  // numbers
{D}+{E}{FS}?	              	{ psh_t(crrs, from_dict(NUM_LIT));} // numbers
{D}*"."{D}+({E})?{FS}?	      { psh_t(crrs, from_dict(NUM_LIT));} // floating numbers
{D}+"."{D}*({E})?{FS}?	      { psh_t(crrs, from_dict(NUM_LIT));} // floating numbers

{ID}[ ]{ID}                   { for (int t : parse_initialization(yytext)) psh_t(crrs, t); }
{ID}("."{ID})*	              { for (int t : parse_reference(yytext)) psh_t(crrs, t); }
<<EOF>>                       { return DONE; }
.                             { 
  if (!strcmp("{", yytext)) {
    no_brack_yet = false;
    brack_cnt++;
  }
    
  if (!strcmp("}", yytext)) 
    brack_cnt--;
    
  psh_t(crrs, from_dict(yytext)); // should be before any modifications because it will add '}' to stream

  if (brack_cnt == 0 && !no_brack_yet) {
    no_brack_yet = true;
    samples.push_back(crrs);
    crrs = SAMPLE(new std::vector<int>());
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

  option = find_arg(argv, argc, "input");
  if (!option.empty()) {
    std::cerr << "Didn't specifie input file name" << std::endl;
    CURRENT_FILE = option;
  }

  option = find_arg(argv, argc, "outfile");
  if (option.empty()) {
    std::cerr << "Didn't receive output file" << std::endl;
    outfile = "test_out.parsed";
  } else {
    outfile += option;
  }

  fill_vocabulary();
  crrs = SAMPLE(new std::vector<int>());
  int res = yylex(); // lexing magic happens here

  save_samples(samples, outfile, vector_length);
  return res;
}