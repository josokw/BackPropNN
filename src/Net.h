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
  double getRecentAverageError() const { return _recentAverageError; }
private:
  std::vector<Neuron::Layer> _layers; // m_layers[layerNum][neuronNum]
  double _error;
  double _recentAverageError;
  static double _recentAverageSmoothingFactor;
};

#endif // NET_H
