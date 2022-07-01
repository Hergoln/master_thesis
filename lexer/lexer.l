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

std::string handleName(std::string yytext) {
  auto it = std::find_if(symtable.begin(), symtable.end(), [&yytext](const Symbol sym){ return yytext.compare(sym.name) == 0; });

  if(it != symtable.end()) {
    return "ID" + (*it).no;
  }

  long no = symtable.size();

  symtable.push_back({yytext, no});
  return "ID" + std::to_string(no);
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

{KWs}                         { printf("%s", yytext);}
{string}                      { printf("STRING_LITERAL");}

0[xX]{H}+{IS}?		            { printf("%s", yytext); } // numbers
0{D}+{IS}?		                { printf("%s", yytext); }  // numbers
{D}+{IS}?		                  { printf("%s", yytext); }  // numbers

{D}+{E}{FS}?	              	{ printf("%s", yytext); } // numbers
{D}*"."{D}+({E})?{FS}?	      { printf("%s", yytext); } // floating numbers
{D}+"."{D}*({E})?{FS}?	      { printf("%s", yytext); } // floating numbers

{ID}[ ]{ID}                   { printf("type variable");} // this rules is here because except string literals, values declarations are the only place "ID ID" construction exists in C

{ID}	                        { std::cout << handleName(yytext); }
<<EOF>>                       { printf("\n"); return DONE; }
.                             { printf("%s", yytext); }
%%


int main() {
  return yylex();
}