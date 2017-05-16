#ifndef OUTPUT_LAYER_HPP
#define OUTPUT_LAYER_HPP

#include "layer.hpp"

class OutputLayer : public Layer
{
public:
  OutputLayer(const std::size_t nodeCount, Layer *prev);
  void setTargets(const std::vector<float> &targets);
  void setTargetBuffer(const cl::Buffer &target);
  cl::Buffer targetBuffer;
  void forward() override;
  void backward() override;
  void update() override;
};

#endif // OUTPUT_LAYER_HPP
