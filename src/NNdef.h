#ifndef NNDEF_H
#define NNDEF_H

#include <functional>
#include <string>
#include <vector>

class Neuron;

/// Shared neural network NN definitions.
namespace nndef {

using connection_t = struct connection {
   double weight;
   double deltaWeight;
};

using topology_t = std::vector<unsigned>;

using action_function_names_t = std::vector<std::string>;
using action_function_t = std::function<double(double)>;

using neurons_layer_t = std::vector<Neuron>;
using neurons_all_layers_t = std::vector<nndef::neurons_layer_t>;

using values_layer_t = std::vector<double>;

using in_out_pair_t = std::pair<nndef::values_layer_t, nndef::values_layer_t>;
using in_out_all_pairs_t = std::vector<nndef::in_out_pair_t>;

inline const nndef::action_function_names_t all_action_function_names{
   "tanh", "sigmoid", "relu", "leaky_relu"};

} // namespace nndef

#endif