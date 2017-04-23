all: main.cpp
	mkdir bin
	g++ -o bin/program main.cpp -lOpenCL

clean:
	rm -rf bin
