#ifndef INPUT_LAYER_HPP
#define INPUT_LAYER_HPP

#include "layer.hpp"

class InputLayer : public Layer
{
public:
  InputLayer(const std::size_t nodeCount, const std::size_t biasCount);
  void setInput(const std::vector<float> &input);
  void setInputBuffer(const cl::Buffer &input);
  void forward() override;
  void backward() override;
  void update() override;
};

#endif // INPUT_LAYER_HPP
