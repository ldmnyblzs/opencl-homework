#include <iostream>
#include <iomanip>
#include <random>
#include <string>


#include "inputlayer.hpp"
#include "outputlayer.hpp"

int main(int argc, char* argv[])
{
  try
    {
      InputLayer input(2, 1);
      Layer hidden(2, 1, &input);
      OutputLayer output(1, &hidden);

      cl_int error;
      float data[] = {1.0f, 1.0f, 1.0f, 0.0f,
		      1.0f, 0.0f, 1.0f, 1.0f,
		      0.0f, 1.0f, 1.0f, 1.0f,
		      0.0f, 0.0f, 1.0f, 0.0f};
      cl::Buffer buffer = Context::getInstance().createBuffer(CL_MEM_READ_WRITE,
							      sizeof(float) * 16,
							      NULL);

      Context::getInstance().writeBuffer(buffer,
					 true,
					 0,
					 sizeof(float) * 16,
					 data);

      cl::Buffer subbuffers[8];
      for (unsigned char i = 0; i < 4; ++i)
	{
	  cl_buffer_region info = {i * 4 * sizeof(float), 3 * sizeof(float)};
	  subbuffers[i] = buffer.createSubBuffer(CL_MEM_READ_WRITE,
						 CL_BUFFER_CREATE_TYPE_REGION,
						 &info);
	  cl_buffer_region info2 = {(i * 4 + 3) * sizeof(float), sizeof(float)};
	  subbuffers[i + 4] = buffer.createSubBuffer(CL_MEM_READ_WRITE,
						     CL_BUFFER_CREATE_TYPE_REGION,
						     &info2);
	}

      std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(4);
      for (std::size_t epoch = 0; epoch < std::stol(argv[1]); epoch++)
	{
	  for (std::size_t index = 0; index < 4; ++index)
	    {
	      input.setInputBuffer(subbuffers[index]);
	      input.forward();
	      output.setTargetBuffer(subbuffers[index + 4]);
	      output.backward();
	      output.update();
	      if (0 == (epoch % 3))
		Context::getInstance().finish();
	    }
	}

      for (std::size_t index = 0; index < 4; ++index)
	{
	  input.setInputBuffer(subbuffers[index]);
	  input.forward();
	  std::cout << data[index * 4] << '\t' << data[index * 4 + 1];
	  for (auto o : output.getOutput())
	    std::cout << '\t' << o;
	  std::cout << std::endl;
	}
    }
  catch (std::exception &e)
    {
      std::cerr << e.what() << std::endl;
    }
  
  return 0;
}
