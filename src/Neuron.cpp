#include "Neuron.h"
#include "ActivationFunctions.h"
#include "NNconfig.h"
#include "NNdef.h"
#include "OSstate.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

std::ostream &operator<<(std::ostream &os, const Neuron &neuron)
{
   OSstate state{os};
   os << "Neuron[" << neuron.myIndex_ << "]  out=" << std::showpos
      << std::setw(6) << std::fixed << std::setprecision(3)
      << neuron.outputVal_;
   if (!neuron.outputWeights_.empty()) {
      os << "  weights=[ ";
      for (const auto &w : neuron.outputWeights_) {
         os << std::setw(6) << std::fixed << std::setprecision(3) << w.weight
            << " ";
      }
      os << "]";
   }
   return os;
}

///< Overall net learning rate, [0.0..1.0]
double Neuron::eta = DEFAULT_ETA;
///< Momentum, multiplier of last deltaWeight, [0.0..1.0]
double Neuron::alpha = DEFAULT_ALPHA;

Neuron::Neuron(std::size_t numOutputs, std::size_t myIndex,
               const std::string &activation_function_name, std::size_t fan_in)
   : outputVal_{0.0}
   , outputWeights_{}
   , myIndex_{myIndex}
   , gradient_{0.0}
   , af_{nn::act_fs[activation_function_name].first}
   , af_derivative_{nn::act_fs[activation_function_name].second}
{
   // He init for the ReLU family, Glorot/Xavier otherwise.
   const bool relu_family = activation_function_name == "relu" or
                            activation_function_name == "leaky_relu";
   for (std::size_t c = 0; c < numOutputs; ++c) {
      outputWeights_.push_back(nndef::connection_t());
      outputWeights_.back().weight =
         relu_family ? nn::he_init(fan_in) : nn::xavier_init(fan_in, numOutputs);
   }
}

void Neuron::updateInputWeights(nndef::neurons_layer_t &prevLayer)
{
   // The weights to be updated are in the Connection container
   // in the neurons in the preceding layer
   for (auto &neuron : prevLayer) {
      double oldDeltaWeight = neuron.outputWeights_[myIndex_].deltaWeight;
      double newDeltaWeight =
         // Individual input, magnified by the gradient and train rate:
         eta * neuron.getOutputVal() * gradient_
         // Also add momentum = a fraction of the previous delta weight;
         + alpha * oldDeltaWeight;
      neuron.outputWeights_[myIndex_].deltaWeight = newDeltaWeight;
      neuron.outputWeights_[myIndex_].weight += newDeltaWeight;
   }
}

double Neuron::sumDOW(const nndef::neurons_layer_t &nextLayer) const
{
   double sum = 0.0;
   // Sum our contributions of the errors at the real nodes we feed
   // (the next layer's bias neuron has no incoming weights).
   for (std::size_t n = 0; n < nextLayer.size() - 1; ++n) {
      sum += outputWeights_[n].weight * nextLayer[n].gradient_;
   }
   return sum;
}

void Neuron::calcHiddenGradients(const nndef::neurons_layer_t &nextLayer)
{
   double dow = sumDOW(nextLayer);
   gradient_ = dow * Neuron::activationFunctionDerivative(outputVal_);
}

void Neuron::calcOutputGradients(double targetVal)
{
   double delta = targetVal - outputVal_;
   gradient_ = delta * Neuron::activationFunctionDerivative(outputVal_);
}

double Neuron::activationFunction(double z)
{
   return af_(z);
}

double Neuron::activationFunctionDerivative(double z)
{
   return af_derivative_(z);
}

void Neuron::feedForward(const nndef::neurons_layer_t &prevLayer)
{
   auto calc_sum = [=, this](double current, const Neuron &neuron) {
      return current +
             (neuron.getOutputVal() * neuron.outputWeights_[myIndex_].weight);
   };

   auto sum =
      std::accumulate(prevLayer.cbegin(), prevLayer.cend(), 0.0, calc_sum);

   outputVal_ = Neuron::activationFunction(sum);
}
