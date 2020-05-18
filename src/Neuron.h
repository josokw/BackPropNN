#ifndef NEURON_H
#define NEURON_H

#include "NNdef.h"

#include <cstdlib>
#include <vector>

class Neuron
{
public:
   using Layer = std::vector<Neuron>;

   Neuron(unsigned numOutputs, unsigned myIndex);
   ~Neuron() = default;

   void setOutputVal(double val) { outputVal_ = val; }
   double getOutputVal(void) const { return outputVal_; }
   void feedForward(const nndef::neurons_layer_t &prevLayer);
   void calcOutputGradients(double targetVal);
   void calcHiddenGradients(const nndef::neurons_layer_t &nextLayer);
   void updateInputWeights(nndef::neurons_layer_t &prevLayer);

private:
   struct Connection {
      double weight;
      double deltaWeight;
   };
   static double eta;   // [0.0..1.0] overall net training rate
   static double alpha; // [0.0..n] multiplier of last weight change (momentum)

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
