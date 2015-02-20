#include "Neuron.h"
#include <cmath>

using namespace std;

double Neuron::eta = 0.15;   // overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0.5;  // momentum, multiplier of last deltaWeight, [0.0..1.0]

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
  : _outputVal {0.0}
  , _outputWeights {}
  , _myIndex {0}
  , _gradient {0.0}
{
  for (unsigned c = 0; c < numOutputs; ++c) {
    _outputWeights.push_back(Connection());
    _outputWeights.back().weight = randomWeight();
  }
  _myIndex = myIndex;
}

void Neuron::updateInputWeights(Neuron::Layer &prevLayer)
{
  // The weights to be updated are in the Connection container
  // in the neurons in the preceding layer
  for (unsigned n = 0; n < prevLayer.size(); ++n) {
    Neuron& neuron = prevLayer[n];
    double oldDeltaWeight = neuron._outputWeights[_myIndex].deltaWeight;
    double newDeltaWeight =
        // Individual input, magnified by the gradient and train rate:
        eta
        * neuron.getOutputVal()
        * _gradient
        // Also add momentum = a fraction of the previous delta weight;
        + alpha
        * oldDeltaWeight;
    neuron._outputWeights[_myIndex].deltaWeight = newDeltaWeight;
    neuron._outputWeights[_myIndex].weight += newDeltaWeight;
  }
}

double Neuron::sumDOW(const Neuron::Layer &nextLayer) const
{
  double sum = 0.0;
  // Sum our contributions of the errors at the nodes we feed.
  for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
    sum += _outputWeights[n].weight * nextLayer[n]._gradient;
  }
  return sum;
}

void Neuron::calcHiddenGradients(const Neuron::Layer &nextLayer)
{
  double dow = sumDOW(nextLayer);
  _gradient = dow * Neuron::transferFunctionDerivative(_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
  double delta = targetVal - _outputVal;
  _gradient = delta * Neuron::transferFunctionDerivative(_outputVal);
}

double Neuron::transferFunction(double x)
{
  // tanh - output range [-1.0..1.0]
  return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
  // tanh derivative
  return 1.0 - x * x;
}

void Neuron::feedForward(const Neuron::Layer &prevLayer)
{
  double sum = 0.0;
  // Sum the previous layer's outputs (which are our inputs)
  // Include the bias node from the previous layer.
  for (unsigned n = 0; n < prevLayer.size(); ++n) {
    sum += prevLayer[n].getOutputVal() *
           prevLayer[n]._outputWeights[_myIndex].weight;
  }
  _outputVal = Neuron::transferFunction(sum);
}
