#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include <string>
#include <cstdint>

#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110
#define CL_HPP_CL_1_2_DEFAULT_BUILD 1
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include <CL/cl2.hpp>

#include "networkexception.hpp"

class Context
{
  cl::Context context;
  std::vector<cl::Device> devices;
  cl::Program program;
  cl::CommandQueue queue;
  Context();
public:
  static Context& getInstance()
  {
    static Context instance;
    return instance;
  }
  Context(const Context &) = delete;
  bool operator=(const Context &) = delete;
  
  cl::Kernel createKernel(const char *name, cl_int *error) const;
  cl::Buffer createBuffer(cl_mem_flags flags,
			  cl::size_type size,
			  void* host_ptr = NULL,
			  cl_int *error = NULL) const;

  cl_int readBuffer(const cl::Buffer &buffer,
		    cl_bool blocking,
		    cl::size_type offset,
		    cl::size_type size,
		    void *ptr,
		    std::vector<cl::Event> *events = NULL,
		    cl::Event *event = NULL) const;
  cl_int writeBuffer(const cl::Buffer &buffer,
		     cl_bool blocking,
		     cl::size_type offset,
		     cl::size_type size,
		     const void *ptr,
		     std::vector<cl::Event> *events = NULL,
		     cl::Event *event = NULL) const;
  cl_int executeKernel(const cl::Kernel &kernel,
		       const cl::NDRange &offset,
		       const cl::NDRange &global,
		       const cl::NDRange &local = cl::NullRange,
		       const std::vector<cl::Event> *events = NULL,
		       cl::Event *event = NULL) const;
  void finish();
};

#endif // CONTEXT_HPP
