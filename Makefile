all: context.cpp layer.cpp inputlayer.cpp outputlayer.cpp main.cpp
	mkdir -p bin
	g++ -o bin/program context.cpp layer.cpp inputlayer.cpp outputlayer.cpp main.cpp -std=c++11 -lOpenCL -O3

clean:
	rm -rf bin

debug: context.cpp layer.cpp inputlayer.cpp outputlayer.cpp main.cpp
	mkdir -p debug
	g++ -o debug/program context.cpp layer.cpp inputlayer.cpp outputlayer.cpp main.cpp -std=c++11 -lOpenCL -g
