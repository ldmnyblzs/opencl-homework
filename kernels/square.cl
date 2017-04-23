__kernel void square(__global float* inputData,
		     __global float* outputData){
  int id = get_global_id(0);
  outputData[id] = inputData[id] * inputData[id];
}
