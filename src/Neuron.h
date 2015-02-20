#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <cstdlib>

class Neuron
{
public:
  using Layer = std::vector<Neuron>;
  Neuron(unsigned numOutputs, unsigned myIndex);
  ~Neuron() = default;
  void setOutputVal(double val) { _outputVal = val; }
  double getOutputVal(void) const { return _outputVal; }
  void feedForward(const Neuron::Layer& prevLayer);
  void calcOutputGradients(double targetVal);
  void calcHiddenGradients(const Neuron::Layer& nextLayer);
  void updateInputWeights(Neuron::Layer& prevLayer);
private:
  struct Connection
  {
    double weight;
    double deltaWeight;
  };
  static double eta;   // [0.0..1.0] overall net training rate
  static double alpha; // [0.0..n] multiplier of last weight change (momentum)
  static double transferFunction(double x);
  static double transferFunctionDerivative(double x);
  static double randomWeight() { return std::rand() / double(RAND_MAX); }
  double sumDOW(const Neuron::Layer& nextLayer) const;
  double _outputVal;
  std::vector<Connection> _outputWeights;
  unsigned _myIndex;
  double _gradient;
};

#endif // NEURON_H
