#include "Net.h"
#include "NNdef.h"
#include "Neuron.h"

#include <cassert>
#include <cmath>
#include <iostream>

double Net::recentAverageSmoothingFactor_ =
   100.0; // Number of training samples to average over

Net::Net(const nndef::topology_t &topology,
         const nndef::action_function_names_t &action_function_names)
   : topology_{topology}
   , action_function_names_{action_function_names}
   , layers_{}
   , RMSerror_{0.0}
   , recentAverageError_{0.5}
{
   std::cout << "NN topology\n";
   auto numLayers = topology.size();
   for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
      layers_.push_back(nndef::neurons_layer_t());
      unsigned numOutputs =
         (layerNum == topology.size() - 1) ? 0 : topology[layerNum + 1];
      // We have a new layer, now fill it with neurons, and
      // add a bias neuron in each layer.
      std::cout << "Layer " << layerNum + 1;
      if (layerNum == 0) {
         std::cout << "                        ";
      } else {
         std::cout << " neurons + bias neuron: ";
      }
      for (unsigned neuronNum = 0; neuronNum <= topology[layerNum];
           ++neuronNum) {
         layers_.back().push_back(
            Neuron{numOutputs, neuronNum, action_function_names_[layerNum]});
         std::cout << '.';
      }
      std::cout << std::endl;
      // Force the bias node's output to 1.0 (it was the last neuron pushed in
      // this layer):
      layers_.back().back().setOutputVal(1.0);
   }
}

void Net::getResults(nndef::values_layer_t &resultVals) const
{
   resultVals.clear();
   for (unsigned n = 0; n < layers_.back().size() - 1; ++n) {
      resultVals.push_back(layers_.back()[n].getOutputVal());
   }
}

void Net::backProp(const nndef::values_layer_t &targetVals)
{
   // Calculate overall net error (RMS of output neuron errors)
   nndef::neurons_layer_t &outputLayer = layers_.back();
   double error = 0.0;

   for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
      double delta = targetVals[n] - outputLayer[n].getOutputVal();
      error += delta * delta;
   }
   error /= outputLayer.size() - 1; // get average error squared
   RMSerror_ = sqrt(error);
   // Implement a recent average measurement
   recentAverageError_ =
      (recentAverageError_ * recentAverageSmoothingFactor_ + RMSerror_) /
      (recentAverageSmoothingFactor_ + 1.0);

   // Calculate output layer gradients
   for (size_t n = 0; auto &neuron : outputLayer) {
      neuron.calcOutputGradients(targetVals[n++]);
   }

   // Calculate hidden layer gradients
   for (unsigned layerNum = layers_.size() - 2; layerNum > 0; --layerNum) {
      nndef::neurons_layer_t &hiddenLayer = layers_[layerNum];
      nndef::neurons_layer_t &nextLayer = layers_[layerNum + 1];

      for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
         hiddenLayer[n].calcHiddenGradients(nextLayer);
      }
   }
   // For all layers from outputs to first hidden layer,
   // update connection weights
   for (auto layerNum = layers_.size() - 1; layerNum > 0; --layerNum) {
      nndef::neurons_layer_t &layer = layers_[layerNum];
      nndef::neurons_layer_t &prevLayer = layers_[layerNum - 1];
      for (unsigned n = 0; n < layer.size() - 1; ++n) {
         layer[n].updateInputWeights(prevLayer);
      }
   }
}

void Net::feedForward(const nndef::values_layer_t &inputVals)
{
   assert(inputVals.size() == layers_[0].size() - 1);

   // Assign (latch) the input values into the input neurons
   for (size_t i = 0; i < inputVals.size(); ++i) {
      layers_[0][i].setOutputVal(inputVals[i]);
   }
   // forward propagate
   for (unsigned layerNum = 1; layerNum < layers_.size(); ++layerNum) {
      nndef::neurons_layer_t &prevLayer = layers_[layerNum - 1];
      for (unsigned n = 0; n < layers_[layerNum].size() - 1; ++n) {
         layers_[layerNum][n].feedForward(prevLayer);
      }
   }
}
