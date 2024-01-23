#ifndef NEURON_H
#define NEURON_H

#include "ActivationFunctions.h"
#include "NNdef.h"

#include <cstdlib>
#include <string>
#include <vector>

/// The class Neuron represents a neuron, every neuron has an index myIndex,
/// a number of outputs, a reference to a chosen activation function and it's
/// derivative.
class Neuron
{
public:
   Neuron(unsigned numOutputs, unsigned myIndex,
          const std::string &action_function_name = "tanh");
   ~Neuron() = default;

   void setOutputVal(double val) { outputVal_ = val; }
   double getOutputVal(void) const { return outputVal_; }
   /// Sums the previous layer's outputs (which are our inputs).
   /// Includes the bias node from the previous layer.
   void feedForward(const nndef::neurons_layer_t &prevLayer);
   void calcOutputGradients(double targetVal);
   void calcHiddenGradients(const nndef::neurons_layer_t &nextLayer);
   void updateInputWeights(nndef::neurons_layer_t &prevLayer);

private:
   /// [0.0..1.0] overall net training rate.
   static double eta;
   // [0.0..n] multiplier of last weight change (momentum).
   static double alpha;

   /// Activation function.
   double activationFunction(double z);
   /// Activation derivative function.
   double activationFunctionDerivative(double z);
   /// For randomly initialisation of the weigths.
   static double randomWeight() { return std::rand() / double(RAND_MAX); }

   double sumDOW(const nndef::neurons_layer_t &nextLayer) const;
   double outputVal_;
   std::vector<nndef::connection_t> outputWeights_;

   size_t myIndex_;
   double gradient_;
   nndef::action_function_t &af_;
   nndef::action_function_t &af_derivative_;
};

#endif // NEURON_H
