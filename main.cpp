// Skeleton.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <vector>
#include <cmath>

#include "util.hpp"

const size_t bufferSize = 1024;

int main()
{
  cl_int err = CL_SUCCESS;

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.size() == 0)
    {
      std::cout << "Unable to find suitable platform." << std::endl;
      return -1;
    }

  cl_context_properties properties[] =
    { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
  cl::Context context(CL_DEVICE_TYPE_GPU, properties);

  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

  std::string programSource = FileToString("kernels/square.cl");
  cl::Program program = cl::Program(context, programSource);
  program.build(devices);

  cl::Kernel kernel(program, "square", &err);

  std::vector<float> hostBuffer;
  for (size_t index = 0; index < bufferSize; ++index)
    {
      hostBuffer.push_back(static_cast<float>(index));
    }

  cl::Buffer clInputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * bufferSize, NULL, &err);
  cl::Buffer clResultBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * bufferSize, NULL, &err);
	
  cl::Event event;
  cl::CommandQueue queue(context, devices[0], 0, &err);
  queue.enqueueWriteBuffer(clInputBuffer, true, 0, sizeof(float) * bufferSize, hostBuffer.data());
  kernel.setArg(0, clInputBuffer);
  kernel.setArg(1, clResultBuffer);
  queue.enqueueNDRangeKernel(kernel,
			     cl::NullRange,
			     cl::NDRange(bufferSize, 1),
			     cl::NullRange,
			     NULL,
			     &event);
  event.wait();

  std::vector<float> resultBuffer(bufferSize);
  queue.enqueueReadBuffer(clResultBuffer, true, 0, sizeof(float) * bufferSize, resultBuffer.data());

  for (size_t index = 0; index < bufferSize; ++index)
    {
      if (resultBuffer[index] != std::pow(hostBuffer[index], 2))
	{
	  std::cout << "Wrong result [" << index << "]: h=" << std::pow(hostBuffer[index], 2) << " g=" << resultBuffer[index] << std::endl;
	  break;
	}
    }

  return 0;
}
