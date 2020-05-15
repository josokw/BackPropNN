#ifndef NET_H
#define NET_H

#include "Neuron.h"

#include <vector>

class Net
{
public:
  Net(const std::vector<unsigned>& topology);
  ~Net() = default;

  void feedForward(const std::vector<double>& inputVals);
  void backProp(const std::vector<double>& targetVals);
  void getResults(std::vector<double>& resultVals) const;
  double getRecentAverageError() const { return recentAverageError_; }

private:
  std::vector<Neuron::Layer> layers_; // m_layers[layerNum][neuronNum]
  double error_;
  double recentAverageError_;
  static double recentAverageSmoothingFactor_;
};

#endif // NET_H
