#include "Net.h"

#include <cassert>
#include <cmath>
#include <iostream>

double Net::recentAverageSmoothingFactor_ = 100.0; // Number of training samples to average over

Net::Net(const std::vector<unsigned>& topology)
  : layers_ {}
  , error_ {0}
  , recentAverageError_ {0.5}
{
  auto numLayers = topology.size();
  for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
    layers_.push_back(Neuron::Layer());
    unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
    // We have a new layer, now fill it with neurons, and
    // add a bias neuron in each layer.
    std::cout << "Neurons ";
    for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
      layers_.back().push_back(Neuron(numOutputs, neuronNum));
      std::cout << '.';
    }
    std::cout << std::endl;
    // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
    layers_.back().back().setOutputVal(1.0);
  }
}

void Net::getResults(std::vector<double>& resultVals) const
{
  resultVals.clear();
  for (unsigned n = 0; n < layers_.back().size() - 1; ++n) {
    resultVals.push_back(layers_.back()[n].getOutputVal());
  }
}

void Net::backProp(const std::vector<double>& targetVals)
{
  // Calculate overall net error (RMS of output neuron errors)
  Neuron::Layer& outputLayer = layers_.back();
  error_ = 0.0;

  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
    double delta = targetVals[n] - outputLayer[n].getOutputVal();
    error_ += delta * delta;
  }
  error_ /= outputLayer.size() - 1; // get average error squared
  error_ = sqrt(error_); // RMS
  // Implement a recent average measurement
  recentAverageError_ =
      (recentAverageError_ * recentAverageSmoothingFactor_ + error_)
      / (recentAverageSmoothingFactor_ + 1.0);

  // Calculate output layer gradients
  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
    outputLayer[n].calcOutputGradients(targetVals[n]);
  }

  // Calculate hidden layer gradients
  for (unsigned layerNum = layers_.size() - 2; layerNum > 0; --layerNum) {
    Neuron::Layer& hiddenLayer = layers_[layerNum];
    Neuron::Layer& nextLayer = layers_[layerNum + 1];

    for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
      hiddenLayer[n].calcHiddenGradients(nextLayer);
    }
  }
  // For all layers from outputs to first hidden layer,
  // update connection weights
  for (auto layerNum = layers_.size() - 1; layerNum > 0; --layerNum) {
    Neuron::Layer& layer = layers_[layerNum];
    Neuron::Layer& prevLayer = layers_[layerNum - 1];
    for (unsigned n = 0; n < layer.size() - 1; ++n) {
      layer[n].updateInputWeights(prevLayer);
    }
  }
}

void Net::feedForward(const std::vector<double>& inputVals)
{
  assert(inputVals.size() == layers_[0].size() - 1);
  
  // Assign (latch) the input values into the input neurons
  for (size_t i = 0; i < inputVals.size(); ++i) {
    layers_[0][i].setOutputVal(inputVals[i]);
  }
  // forward propagate
  for (unsigned layerNum = 1; layerNum < layers_.size(); ++layerNum) {
    Neuron::Layer& prevLayer = layers_[layerNum - 1];
    for (unsigned n = 0; n < layers_[layerNum].size() - 1; ++n) {
      layers_[layerNum][n].feedForward(prevLayer);
    }
  }
}
