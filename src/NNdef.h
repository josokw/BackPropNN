#ifndef NNDEF_H
#define NNDEF_H

#include <vector>

class Neuron;

/// Shared neural network NN definitions.
namespace nndef {

using topology_t = std::vector<unsigned>;

using neurons_layer_t = std::vector<Neuron>;
using neurons_all_layers_t = std::vector<nndef::neurons_layer_t>;

using values_layer_t = std::vector<double>;

using in_out_pair_t = std::pair<nndef::values_layer_t, nndef::values_layer_t>;
using in_out_all_pairs_t = std::vector<nndef::in_out_pair_t>;

} // namespace nndef

#endif