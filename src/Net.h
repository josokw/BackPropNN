#ifndef NET_H
#define NET_H

#include "NNdef.h"
#include "Neuron.h"

/// The class Net represents a layered neural network (NN).
/// Layers: 1 input layer, >=1 hidden layers, 1 output layer.
/// The NN is dynamically build by the topology data.
class Net
{
public:
   Net(const nndef::topology_t &topology);
   ~Net() = default;

   const auto &topology() const { return topology_; }
   void feedForward(const nndef::values_layer_t &inputVals);
   void backProp(const nndef::values_layer_t &targetVals);
   void getResults(nndef::values_layer_t &resultVals) const;
   double getRecentAverageError() const { return recentAverageError_; }

private:
   const nndef::topology_t &topology_;
   nndef::neurons_all_layers_t layers_; // m_layers[layerNum][neuronNum]
   double RMSerror_;
   double recentAverageError_;
   static double recentAverageSmoothingFactor_;
};

#endif // NET_H
