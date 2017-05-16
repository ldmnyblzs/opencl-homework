#include "outputlayer.hpp"

OutputLayer::OutputLayer(const std::size_t nodeCount, Layer* prev) : Layer(nodeCount, 0, prev, "sigmoid", "backprop_output", "update")
{
  cl_int error;
  this->targetBuffer = Context::getInstance().createBuffer(CL_MEM_READ_ONLY,
							   sizeof(float) * this->nodeCount,
							   NULL,
							   &error);
}

void OutputLayer::forward()
{
  this->forwardKernel.setArg(0, this->prev->outputBuffer);
  this->forwardKernel.setArg(1, this->prev->nodeCount + this->prev->biasCount);
  this->forwardKernel.setArg(2, this->weightBuffer);
  this->forwardKernel.setArg(3, this->outputBuffer);

  Context::getInstance().executeKernel(this->forwardKernel,
				       cl::NullRange,
				       cl::NDRange(this->nodeCount),
				       cl::NullRange);
}

void OutputLayer::backward()
{
  this->backwardKernel.setArg(0, this->outputBuffer);
  this->backwardKernel.setArg(1, this->targetBuffer);
  this->backwardKernel.setArg(2, this->deltaBuffer);

  Context::getInstance().executeKernel(this->backwardKernel,
				       cl::NullRange,
				       cl::NDRange(this->nodeCount),
				       cl::NullRange);
  this->prev->backward();
}

void OutputLayer::update()
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

void OutputLayer::setTargets(const std::vector<float> &target)
{
  if (target.size() != (this->nodeCount + this->biasCount))
    throw NetworkException("Target count doesn't match node count!");

  Context::getInstance().writeBuffer(this->targetBuffer,
				     true,
				     0,
				     sizeof(float) * (this->nodeCount + this->biasCount),
				     target.data());
}

void OutputLayer::setTargetBuffer(const cl::Buffer &target)
{
  this->targetBuffer = target;
}
