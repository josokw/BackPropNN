#ifndef NEURON_H
#define NEURON_H

#include "NNdef.h"

#include <cstdlib>
#include <vector>

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
   struct Connection {
      double weight;
      double deltaWeight;
   };
   /// [0.0..1.0] overall net training rate.
   static double eta;
   // [0.0..n] multiplier of last weight change (momentum)
   static double alpha; 

   static double transferFunction(double x);
   static double transferFunctionDerivative(double x);
   static double randomWeight() { return std::rand() / double(RAND_MAX); }

   double sumDOW(const nndef::neurons_layer_t &nextLayer) const;
   double outputVal_;
   std::vector<Connection> outputWeights_;

   size_t myIndex_;
   double gradient_;
};

#endif // NEURON_H
