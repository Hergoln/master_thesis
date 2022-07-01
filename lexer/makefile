lexer : lexer.o global.h
	g++ -o lexer lexer.o -lfl

lexer.o : lexer.cpp global.h
	g++ -c lexer.cpp

lexer.cpp : lexer.l global.h
	flex -o lexer.cpp lexer.l

clean : 
	rm lexer.o
	rm lexer
	rm lexer.cpp
