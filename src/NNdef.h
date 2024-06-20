#ifndef NNDEF_H
#define NNDEF_H

#include <functional>
#include <map>
#include <string>
#include <vector>

class Neuron;
class TrainingData;

/// Shared neural network NN definitions.
namespace nndef {

using connection_t = struct connection {
   double weight;
   double deltaWeight;
};

using topology_t = std::vector<unsigned>;

using activation_function_names_t = std::vector<std::string>;
using activation_function_t = std::function<double(double)>;

using neurons_layer_t = std::vector<Neuron>;
using neurons_all_layers_t = std::vector<nndef::neurons_layer_t>;

using values_layer_t = std::vector<double>;

using in_out_pair_t = std::pair<nndef::values_layer_t, nndef::values_layer_t>;
using in_out_all_pairs_t = std::vector<nndef::in_out_pair_t>;

using semantic_activation_function_t =
   std::function<void(std::stringstream &, TrainingData &)>;
using semantic_activations_t =
   std::map<std::string, nndef::semantic_activation_function_t>;

inline const nndef::activation_function_names_t all_activation_function_names{
   "tanh", "sigmoid", "relu", "leaky_relu"};

} // namespace nndef

#endif