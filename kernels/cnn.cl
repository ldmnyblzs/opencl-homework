kernel void random(const uint2 random,
		   global float* randomData)
{
  const size_t index = get_global_id(0);
  const uint seed = index * 128 + random.x;
  const uint t = seed ^ (seed << 11);
  const uint result = random.y ^ (random.y >> 19) ^ (t ^ (t >> 8));
  randomData[index] = 1.0f - (float) (result % 128) / 64.0f;
}

/*__kernel void convolution(const __global int* inputData,
			  __constant int* filterData,
			  const int filterSize,
			  __global int* outputData,
			  const width) {
  const size_t outWidth = get_global_size(0);
  const size_t outColumn = get_global_id(0);
  const size_t outRow = get_global_id(1);
  const size_t outIndex = outRow * outWidth + outColumn;

  int sum = 0;
  for (int column = 0; column < filterSize; ++column)
    {
      for (int row = 0; row < filterSize; ++row)
	{
	  const int inIndex = (outRow + row) * width + (outColumn + column);
	  const int filterIndex = column * filterSize + row;
	  sum += inputData[inIndex] * filterData[filterIndex];
	}
    }
  outputData[outIndex] = max(0, sum);
}

__kernel void pooling(const __global int* inData,
		      const size_t poolSize,
		      __global int* outData)
{
  const size_t outWidth = get_global_size(0);
  const size_t outColumn = get_global_id(0);
  const size_t outRow = get_global_id(1);
  const size_t outIndex = outRow * outWidth + outColumn;
  const size_t inWidth = outWidth * poolSize;
  const size_t leftColumn = outColumn * poolSize;
  const size_t topRow = outRow * poolSize;

  int max = 0;

  for (size_t column = 0; column < poolSize; ++column)
    {
      for (size_t row = 0; row < poolSize; ++row)
	{
	  const size_t inIndex = (topRow + row) * inWidth + (leftColumn + column);
	  max = max(max, inData[inIndex]);
	}
    }
  outData[outIndex] = max;
  }*/

kernel void
sigmoid(constant float* inputData,
	const unsigned long inputCount,
	constant float* weights,
	global float* outputData)
{
  const size_t outputIndex = get_global_id (0);
  const size_t outputCount = get_global_size (0);

  float sum = 0;
  for(size_t inputIndex = 0; inputIndex < inputCount; ++inputIndex)
      sum += inputData[inputIndex] * weights[inputIndex * outputCount + outputIndex];
  outputData[outputIndex] = 1.0f / (1.0f + exp(-1.0f * sum));
}

// calculate node deltas on the output layer
kernel void
backprop_output(constant float* outputs,
		constant float* targets,
		global float* deltas)
{
  const size_t outputIndex = get_global_id (0);
  const float out = outputs[outputIndex];
  deltas[outputIndex] = (out - targets[outputIndex]) * out * (1.0f - out);
}

// calculate node deltas on the hidden layers
kernel void
backprop_hidden(constant float* outputs,
		const unsigned long nextCount,
		constant float* nextDeltas,
		constant float* nextWeights,
		global float* deltas)
{
  const size_t outputIndex = get_global_id (0);
  const float out = outputs[outputIndex];
  
  float sum = 0;
  for (size_t nextIndex = 0; nextIndex < nextCount; ++nextIndex)
    sum += nextDeltas[nextIndex] * nextWeights[outputIndex * nextCount + nextIndex];

  deltas[outputIndex] = sum * out * (1.0f - out);
}

kernel void
update(global float* weights,
       constant float* deltas,
       constant float* inputs)
{
  const size_t deltaIndex = get_global_id (0);
  const size_t inputIndex = get_global_id (1);
  const size_t deltaCount = get_global_size (0);

  weights[inputIndex * deltaCount + deltaIndex] -= inputs[inputIndex] * deltas[deltaIndex];
}
