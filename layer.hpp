#ifndef LAYER_HPP
#define LAYER_HPP

#include "context.hpp"

class Layer
{
public:
  Layer(const std::size_t nodeCount,
	const std::size_t biasCount,
	Layer *prev);
  virtual ~Layer() {}
  Layer(const std::size_t nodeCount,
	const std::size_t biasCount,
	Layer *prev,
	const char* forwardName,
	const char* backwardName,
	const char* updateName);
  virtual void forward();
  virtual void backward();
  virtual void update();
  std::vector<float> getOutput();
  std::vector<float> getWeights();
  
  cl::Kernel forwardKernel;
  cl::Kernel backwardKernel;
  cl::Kernel updateKernel;

  cl::Buffer weightBuffer;
  cl::Buffer outputBuffer;
  cl::Buffer deltaBuffer;

  std::size_t nodeCount;
  std::size_t biasCount;
  Layer *prev;
  Layer *next;
};

#endif // LAYER_HPP
