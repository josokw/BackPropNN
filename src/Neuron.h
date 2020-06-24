#ifndef NEURON_H
#define NEURON_H

#include "NNdef.h"

#include <cstdlib>
#include <vector>

/// The class Neuron represents a neuron, every nueron has an index myIndex and
/// has a number of outputs.
class Neuron
{
public:
   Neuron(unsigned numOutputs, unsigned myIndex);
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

   /// Hyperbolic tangent activation function.
   static double transferFunction(double x);
   /// Hyperbolic tangent activation derivative function.
   static double transferFunctionDerivative(double x);
   /// For initialisation of the weigths.
   static double randomWeight() { return std::rand() / double(RAND_MAX); }

   double sumDOW(const nndef::neurons_layer_t &nextLayer) const;
   double outputVal_;
   std::vector<nndef::connection_t> outputWeights_;

   size_t myIndex_;
   double gradient_;
};

#endif // NEURON_H
