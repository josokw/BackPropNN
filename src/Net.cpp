#include "Net.h"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace std;

double Net::_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

Net::Net(const vector<unsigned> &topology)
  : _layers {}
  , _error {0}
  , _recentAverageError {0}
{
  unsigned numLayers = topology.size();
  for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
    _layers.push_back(Neuron::Layer());
    unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
    // We have a new layer, now fill it with neurons, and
    // add a bias neuron in each layer.
    for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
      _layers.back().push_back(Neuron(numOutputs, neuronNum));
      cout << "Made a Neuron!" << endl;
    }
    // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
    _layers.back().back().setOutputVal(1.0);
  }
}

void Net::getResults(vector<double> &resultVals) const
{
  resultVals.clear();
  for (unsigned n = 0; n < _layers.back().size() - 1; ++n) {
    resultVals.push_back(_layers.back()[n].getOutputVal());
  }
}

void Net::backProp(const vector<double> &targetVals)
{
  // Calculate overall net error (RMS of output neuron errors)
  Neuron::Layer& outputLayer = _layers.back();
  _error = 0.0;

  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
    double delta = targetVals[n] - outputLayer[n].getOutputVal();
    _error += delta * delta;
  }
  _error /= outputLayer.size() - 1; // get average error squared
  _error = sqrt(_error); // RMS
  // Implement a recent average measurement
  _recentAverageError =
      (_recentAverageError * _recentAverageSmoothingFactor + _error)
      / (_recentAverageSmoothingFactor + 1.0);

  // Calculate output layer gradients
  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
    outputLayer[n].calcOutputGradients(targetVals[n]);
  }

  // Calculate hidden layer gradients
  for (unsigned layerNum = _layers.size() - 2; layerNum > 0; --layerNum) {
    Neuron::Layer& hiddenLayer = _layers[layerNum];
    Neuron::Layer& nextLayer = _layers[layerNum + 1];

    for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
      hiddenLayer[n].calcHiddenGradients(nextLayer);
    }
  }
  // For all layers from outputs to first hidden layer,
  // update connection weights
  for (unsigned layerNum = _layers.size() - 1; layerNum > 0; --layerNum) {
    Neuron::Layer& layer = _layers[layerNum];
    Neuron::Layer& prevLayer = _layers[layerNum - 1];
    for (unsigned n = 0; n < layer.size() - 1; ++n) {
      layer[n].updateInputWeights(prevLayer);
    }
  }
}

void Net::feedForward(const vector<double> &inputVals)
{
  assert(inputVals.size() == _layers[0].size() - 1);
  // Assign (latch) the input values into the input neurons
  for (unsigned i = 0; i < inputVals.size(); ++i) {
    _layers[0][i].setOutputVal(inputVals[i]);
  }
  // forward propagate
  for (unsigned layerNum = 1; layerNum < _layers.size(); ++layerNum) {
    Neuron::Layer& prevLayer = _layers[layerNum - 1];
    for (unsigned n = 0; n < _layers[layerNum].size() - 1; ++n) {
      _layers[layerNum][n].feedForward(prevLayer);
    }
  }
}
