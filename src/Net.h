#ifndef NET_H
#define NET_H

#include "Neuron.h"

#include <vector>

/// The class Net represents a layered neural network (NN).
/// The NN is configured by the topology data.
class Net
{
public:
   using layer_t = std::vector<double>;

   Net(const std::vector<unsigned> &topology);
   ~Net() = default;

   const auto &topology() const { return topology_; }
   void feedForward(const layer_t &inputVals);
   void backProp(const layer_t &targetVals);
   void getResults(layer_t &resultVals) const;
   double getRecentAverageError() const { return recentAverageError_; }

private:
   const std::vector<unsigned> &topology_;
   std::vector<Neuron::Layer> layers_; // m_layers[layerNum][neuronNum]
   double error_;
   double recentAverageError_;
   static double recentAverageSmoothingFactor_;
};

#endif // NET_H
