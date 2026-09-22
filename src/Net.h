#ifndef NET_H
#define NET_H

#include "NNdef.h"
#include "Neuron.h"

/// The class Net represents a layered neural network (NN).
/// Layers: 1 input layer, >=1 hidden layers, 1 output layer.
/// The NN is dynamically build by the topology and action function names data.
class Net
{
   friend std::ostream &operator<<(std::ostream &os, const Net &net);

public:
   Net(const nndef::topology_t &topology,
       const nndef::activation_function_names_t &activation_function_names);
   ~Net() = default;

   [[nodiscard]] const auto &topology() const { return topology_; }
    ///< The neuron layers (layer 0 = inputs ... layer n = outputs).
    [[nodiscard]] const auto &layers() const { return layers_; }
   void feedForward(const nndef::values_layer_t &inputVals);
   void backProp(const nndef::values_layer_t &targetVals);
   void getResults(nndef::values_layer_t &resultVals) const;
   [[nodiscard]] double getRecentAverageError() const
   {
      return recentAverageError_;
   }

private:
   nndef::topology_t topology_;
   nndef::activation_function_names_t activation_function_names_;
   nndef::neurons_all_layers_t layers_; // layers_[layerNum][neuronNum]
   double RMSerror_;
   double recentAverageError_;
   static double recentAverageSmoothingFactor_;
};

#endif // NET_H
