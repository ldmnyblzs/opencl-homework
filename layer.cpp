#include "layer.hpp"

#include <random>

Layer::Layer(const std::size_t nodeCount,
	     const std::size_t biasCount,
	     Layer* prev,
	     const char* forwardName,
	     const char* backwardName,
	     const char* updateName)
  : nodeCount(nodeCount), biasCount(biasCount), prev(prev)
{
  if (NULL != this->prev)
    this->prev->next = this;
  
  cl_int error;
  if (NULL != forwardName)
    this->forwardKernel = Context::getInstance().createKernel(forwardName, &error);
  if (NULL != backwardName)
    this->backwardKernel = Context::getInstance().createKernel(backwardName, &error);
  if (NULL != updateName)
    this->updateKernel = Context::getInstance().createKernel(updateName, &error);

  if (NULL != this->prev)
    {
      static std::default_random_engine generator;
      static std::uniform_int_distribution<unsigned int> distribution;
      cl_uint2 randomInts = {distribution(generator), distribution(generator)};
      
      this->weightBuffer = Context::getInstance().createBuffer(CL_MEM_READ_WRITE,
							       sizeof(float) * this->nodeCount * (this->prev->nodeCount + this->prev->biasCount),
							       NULL,
							       &error);
      auto random = Context::getInstance().createKernel("random", &error);
      random.setArg(0, randomInts);
      random.setArg(1, this->weightBuffer);
      Context::getInstance().executeKernel(random,
					   cl::NullRange,
					   cl::NDRange(this->nodeCount * (this->prev->nodeCount + this->prev->biasCount)),
					   cl::NullRange);
    }

  std::vector<float> ones(this->nodeCount + this->biasCount, 1.0f);
  this->outputBuffer = Context::getInstance().createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
							   sizeof(float) * (this->nodeCount + this->biasCount),
							   ones.data(),
							   &error);

  this->deltaBuffer = Context::getInstance().createBuffer(CL_MEM_READ_WRITE,
							  sizeof(float) * (this->nodeCount),
							  NULL,
							  &error);
}

Layer::Layer(const std::size_t nodeCount, const std::size_t biasCount, Layer* prev)
  : Layer(nodeCount, biasCount, prev, "sigmoid", "backprop_hidden", "update")
{
}

void Layer::forward()
{
  this->forwardKernel.setArg(0, this->prev->outputBuffer);
  this->forwardKernel.setArg(1, this->prev->nodeCount + this->prev->biasCount);
  this->forwardKernel.setArg(2, this->weightBuffer);
  this->forwardKernel.setArg(3, this->outputBuffer);

  Context::getInstance().executeKernel(this->forwardKernel,
				       cl::NullRange,
				       cl::NDRange(this->nodeCount),
				       cl::NullRange);
  this->next->forward();
}

void Layer::backward()
{
  this->backwardKernel.setArg(0, this->outputBuffer);
  this->backwardKernel.setArg(1, this->next->nodeCount);
  this->backwardKernel.setArg(2, this->next->deltaBuffer);
  this->backwardKernel.setArg(3, this->next->weightBuffer);
  this->backwardKernel.setArg(4, this->deltaBuffer);

  Context::getInstance().executeKernel(this->backwardKernel,
				       cl::NullRange,
				       cl::NDRange(this->nodeCount),
				       cl::NullRange);
  this->prev->backward();
}

void Layer::update()
{
  this->updateKernel.setArg(0, this->weightBuffer);
  this->updateKernel.setArg(1, this->deltaBuffer);
  this->updateKernel.setArg(2, this->prev->outputBuffer);

  Context::getInstance().executeKernel(this->updateKernel,
				       cl::NullRange,
				       cl::NDRange(this->nodeCount, this->prev->nodeCount + this->prev->biasCount),
				       cl::NullRange);
  this->prev->update();
}

std::vector<float> Layer::getOutput()
{
  std::vector<float> result(this->nodeCount + this->biasCount);
  Context::getInstance().readBuffer(this->outputBuffer,
				    true,
				    0,
				    sizeof(float) * (this->nodeCount + this->biasCount),
				    result.data());
  return result;
}

std::vector<float> Layer::getWeights()
{
  std::vector<float> result(this->nodeCount * (this->prev->nodeCount + this->prev->biasCount));
  Context::getInstance().readBuffer(this->weightBuffer,
				    true,
				    0,
				    sizeof(float) * this->nodeCount * (this->prev->nodeCount + this->prev->biasCount),
				    result.data());
  return result;
}
