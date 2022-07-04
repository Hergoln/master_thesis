%{
#include "global.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>

int lineno = 1;
int KEYWORD;
int NUMBER;
int CHAR;
int FLOAT;
int STRING_LITERAL;
int ID;
int DONE;

std::string NUM_LIT = "NUM";
std::string STR_LIT = "STR";

extern std::stringstream ss;
extern std::vector<Symbol> symtable;

std::string handle_name(const std::string yytext, const std::string prefix) {
  if (std::find(keywords.begin(), keywords.end(), yytext) != keywords.end()) {
    return yytext;
  }
  auto it = std::find_if(symtable.begin(), symtable.end(), [&yytext](const Symbol sym){ return yytext.compare(sym.name) == 0; });

  if(it != symtable.end()) {
    return prefix + std::to_string((*it).no);
  }

  long no = symtable.size();

  symtable.push_back({yytext, prefix, no});
  return prefix + std::to_string(no);
}

std::string parse_initialization(const std::string yytext) {
  char del = ' ';
  std::vector<std::string> parsed;
  std::stringstream ss(yytext);
  std::string item, result;

  while(getline(ss, item, del)) parsed.push_back(item);
  parsed[0] = handle_name(parsed[0], "TP");
  parsed[1] = handle_name(parsed[1], "ID");

  return parsed[0] + del + parsed[1];
}

std::string parse_reference(const std::string yytext) {
  char del = '.';
  std::vector<std::string> parsed;
  std::stringstream ss(yytext);
  std::string item, result;

  while(getline(ss, item, del)) parsed.push_back(item);

  result = handle_name(parsed[0], "ID");
  for (int i = 1; i < parsed.size(); ++i) {
    result += ".RF" + std::to_string(i - 1);
  }
  
  return result;
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
\r?\n                         lineno++;
{includ}                      {}
{comment}                     {}
{delim}                       {}

{string}                      { ss << STR_LIT;}

0[xX]{H}+{IS}?		            { ss << NUM_LIT; } // numbers
0{D}+{IS}?		                { ss << NUM_LIT; }  // numbers
{D}+{IS}?		                  { ss << NUM_LIT; }  // numbers
{D}+{E}{FS}?	              	{ ss << NUM_LIT; } // numbers
{D}*"."{D}+({E})?{FS}?	      { ss << NUM_LIT; } // floating numbers
{D}+"."{D}*({E})?{FS}?	      { ss << NUM_LIT; } // floating numbers

{ID}[ ]{ID}                   { ss << parse_initialization(yytext); }
{ID}("."{ID})*	              { ss << parse_reference(yytext); }
<<EOF>>                       { ss << std::endl; return DONE; }
.                             { ss << yytext; }
%%


int main(int argc, char* argv[]) {
  std::string outfile;
  std::ofstream ofs;
  if (argc <= 1) {
    std::cerr << "Didn't receive output file" << std::endl;
    outfile = "test_out.parsed";
  } else {
    outfile += argv[1];
  }

  int res = yylex();
  ofs.open(outfile, std::ofstream::out);
  ofs << ss.str();
  ofs.close();
  std::cout << "Write to file: " << outfile << std::endl;
  return res;
}