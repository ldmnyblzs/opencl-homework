#include "inputlayer.hpp"

InputLayer::InputLayer(const std::size_t nodeCount, const std::size_t biasCount)
  : Layer(nodeCount, biasCount, NULL)
{
}

void InputLayer::forward()
{
  this->next->forward();
}

void InputLayer::backward()
{
}
void InputLayer::update()
{
}

void InputLayer::setInput(const std::vector<float> &input)
{
  if (input.size() != (this->nodeCount + this->biasCount))
    throw NetworkException("Input size doesn't match the input layer's node count!");

  Context::getInstance().writeBuffer(this->outputBuffer,
				     true,
				     0,
				     sizeof(float) * (this->nodeCount + this->biasCount),
				     input.data());
}

void InputLayer::setInputBuffer(const cl::Buffer &input)
{
  this->outputBuffer = input;
}
