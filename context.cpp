#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

#include "context.hpp"
#include "networkexception.hpp"

Context::Context()
{
  cl_int err = CL_SUCCESS;

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.size() == 0)
    throw NetworkException("No platform available!");

  cl_context_properties properties[] =
    { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };

  this->context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
  this->devices = this->context.getInfo<CL_CONTEXT_DEVICES>();

  const char* path = "../kernels/cnn.cl";
  std::ifstream file(path, std::ios::in | std::ios::binary);
  std::ostringstream source;
  source << file.rdbuf();
  file.close();

  try
    {
      this->program = cl::Program(this->context, source.str());
      this->program.build(this->devices);
    }
  catch(cl::Error &error)
    {
      std::cerr << "Build error" << std::endl;
      std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
      exit(1);
    }

  this->queue = cl::CommandQueue(this->context, devices[0]);
}

cl::Kernel Context::createKernel(const char* name, cl_int *error) const
{
  return cl::Kernel(this->program, name, error);
}

cl::Buffer Context::createBuffer(cl_mem_flags flags,
				 cl::size_type size,
				 void* host_ptr,
				 cl_int *error) const
{
  return cl::Buffer(this->context, flags, size, host_ptr, error);
}

cl_int Context::readBuffer(const cl::Buffer &buffer,
			   cl_bool blocking,
			   cl::size_type offset,
			   cl::size_type size,
			   void *ptr,
			   std::vector<cl::Event> *events,
			   cl::Event *event) const
{
  return this->queue.enqueueReadBuffer(buffer, blocking, offset, size, ptr, events, event);
}

cl_int Context::writeBuffer(const cl::Buffer &buffer,
			    cl_bool blocking,
			    cl::size_type offset,
			    cl::size_type size,
			    const void *ptr,
			    std::vector<cl::Event> *events,
			    cl::Event *event) const
{
  return this->queue.enqueueWriteBuffer(buffer, blocking, offset, size, ptr, events, event);
}

cl_int Context::executeKernel(const cl::Kernel &kernel,
			      const cl::NDRange &offset,
			      const cl::NDRange &global,
			      const cl::NDRange &local,
			      const std::vector<cl::Event> *events,
			      cl::Event *event) const
{
  return this->queue.enqueueNDRangeKernel(kernel, offset, global, local, events, event);
}

void Context::finish()
{
  this->queue.flush();
}
