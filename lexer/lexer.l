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

// options
int VECTOR_LENGTH;
std::string INPUT = "NOT_SPECIFIED";
std::string OUTPUT_DIR = ".";

void print_help() {
  std::cout << "inp:\n\tfile that is being processed, actaul stream\n\tshould be passed as stream" <<
   "'< file'" << std::endl << std::endl;
  std::cout << "out:\n\tdirectory that should be place for output files" << std::endl << std::endl;
  std::cout << "vector-length:\n\tlength of the output vectort. If out\n\tvector for a function" <<
  " from given file is" << 
  "\n\tlonger than vector-length than the rest\n\twill be filled with '0' (EMPTY token)" << std::endl;
}

void prepare_options(int argc, char** argv) {
  std::string option = find_arg(argv, argc, "vector-length");
  if(!option.empty()) {
    VECTOR_LENGTH = stoi(option);
  } else {
    VECTOR_LENGTH = DEFAULT_VECTOR_LENGTH;
  }

  option = find_arg(argv, argc, "inp");
  if (!option.empty()) {
    INPUT = option;
  } else {
    std::cerr << "Didn't specifie input file name" << std::endl;
  }

  option = find_arg(argv, argc, "out");
  if (!option.empty()) {
   OUTPUT_DIR = option;
  } else {
    std::cerr << "Didn't specifie output directory" << std::endl;
  }
}

std::string compose_outfile(int f_no, std::string input, std::string out_dir) {
  return out_dir + "/" + input + std::to_string(f_no) + ".parsed";
}

void save_samples(std::vector<SAMPLE> samples, std::string input, std::string output_dir, int vector_len) {
  std::ofstream ofs;
  std::ofstream too_long_ofs;

  too_long_ofs.open(DEFAULT_TOO_LONG_FILE, std::ofstream::app);

  for (int i =0; i < samples.size(); ++i) {
    SAMPLE sample = samples[i];

    if (sample->size() > vector_len) {
      too_long_ofs << input << ": function no.: " << i << std::endl;
      continue;
    }
    ofs.open(compose_outfile(i, input, output_dir), std::ofstream::out);
    // stream tokens into ofstream
    for (int t : *sample) {
      ofs << std::to_string(t) << " ";
    }

    // fill with EMPTY
    for(int i = 0; i < vector_len - sample->size(); ++i)
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
  if (!find_single_arg(argv, argc, "--help").empty()) {
    print_help();
    return 0;
  }

  prepare_options(argc, argv);
  fill_vocabulary();
  crrs = SAMPLE(new std::vector<int>());
  int res = yylex(); // lexing magic happens here
  save_samples(samples, INPUT, OUTPUT_DIR, VECTOR_LENGTH);
  return res;
}