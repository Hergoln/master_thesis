all: decoder lexer

decoder : decoder.o global.h parsingfunctions.o
	g++ -o decoder decoder.o parsingfunctions.o -lfl

decoder.o : global.h
	g++ -c decoder.cpp

lexer : lexer.o global.h parsingfunctions.o
	g++ -o lexer lexer.o parsingfunctions.o -lfl

lexer.o : lexer.cpp global.h
	g++ -c lexer.cpp

lexer.cpp : lexer.l global.h
	flex -o lexer.cpp lexer.l

parsingfunctions.o : parsingfunctions.cpp global.h
	g++ -c parsingfunctions.cpp

clean : 
	-rm decoder.o
	-rm parsingfunctions.o
	-rm lexer.o
	-rm lexer.cpp
	-rm lexer
	-rm decoder