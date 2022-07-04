%{
#include "global.h"
#include <iostream>
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

extern std::vector<Symbol> symtable;

std::string handle_name(const std::string yytext, const std::string prefix) {
  auto it = std::find_if(symtable.begin(), symtable.end(), [&yytext](const Symbol sym){ return yytext.compare(sym.name) == 0; });

  if(it != symtable.end()) {
    return prefix + std::to_string((*it).no);
  }

  long no = symtable.size();

  symtable.push_back({yytext, prefix, no});
  return "ID" + std::to_string(no);
}

std::string parse_initialization(const std::string yytext, char delimiter) {
  std::vector<std::string> parsed;
  std::stringstream ss(yytext);
  std::string item, result;

  while(getline(ss, item, delimiter)) parsed.push_back(item);
  parsed[0] = handle_name(parsed[0], "type");
  parsed[1] = handle_name(parsed[1], "ID");

  return parsed[0] + delimiter + parsed[1];
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

KWs     "return"|"auto"|"break"|"case"|"char"|"const"|"continue"|"default"|"do"|"double"|"else"|"enum"|"extern"|"float"|"for"|"goto"|"if"|"int"|"long"|"register"|"return"|"short"|"signed"|"sizeof"|"static"|"struct"|"switch"|"typedef"|"union"|"unsigned"|"void"|"volatile"|"while"|"true"|"false"
ID       {L}({L}|{D})*

%%
\r?\n                         lineno++;
{includ}                      {}
{comment}                     {}
{delim}                       {}

{KWs}                         { std::cout <<  yytext;}
{string}                      { std::cout << "STRING_LITERAL";}

0[xX]{H}+{IS}?		            { std::cout << yytext; } // numbers
0{D}+{IS}?		                { std::cout << yytext; }  // numbers
{D}+{IS}?		                  { std::cout << yytext; }  // numbers

{D}+{E}{FS}?	              	{ std::cout << yytext; } // numbers
{D}*"."{D}+({E})?{FS}?	      { std::cout << yytext; } // floating numbers
{D}+"."{D}*({E})?{FS}?	      { std::cout << yytext; } // floating numbers

{ID}[ ]{ID}                   { std::cout << parse_initialization(yytext, ' '); }

{ID}	                        { std::cout << handle_name(yytext, "ID"); }
<<EOF>>                       { std::cout << std::endl; return DONE; }
.                             { std::cout << yytext; }
%%


int main() {
  return yylex();
}