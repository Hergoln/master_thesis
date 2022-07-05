%{
#include "global.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>

int DONE;

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
\r?\n                         {};
{includ}                      {}
{comment}                     {}
{delim}                       {}

{string}                      { ss << from_dict(STR_LIT);}

0[xX]{H}+{IS}?		            { ss << from_dict(NUM_LIT); } // numbers
0{D}+{IS}?		                { ss << from_dict(NUM_LIT); }  // numbers
{D}+{IS}?		                  { ss << from_dict(NUM_LIT); }  // numbers
{D}+{E}{FS}?	              	{ ss << from_dict(NUM_LIT); } // numbers
{D}*"."{D}+({E})?{FS}?	      { ss << from_dict(NUM_LIT); } // floating numbers
{D}+"."{D}*({E})?{FS}?	      { ss << from_dict(NUM_LIT); } // floating numbers

{ID}[ ]{ID}                   { ss << parse_initialization(yytext); }
{ID}("."{ID})*	              { ss << parse_reference(yytext); }
<<EOF>>                       { ss << std::endl; return DONE; }
.                             { ss << from_dict(yytext); }
%%

std::stringstream ss;

int main(int argc, char* argv[]) {
  std::string outfile;
  std::ofstream ofs; 

  if (argc <= 1) {
    std::cerr << "Didn't receive output file" << std::endl;
    outfile = "test_out.parsed";
  } else {
    outfile += argv[1];
  }

  fill_vocabulary();
  int res = yylex();
  ofs.open(outfile, std::ofstream::out);
  ofs << ss.str();
  ofs.close();
  std::cout << "Write to file: " << outfile << std::endl;
  return res;
}