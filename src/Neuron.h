#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <cstdlib>

struct Connection
{
  double weight;
  double deltaWeight;
};

class Neuron;
typedef std::vector<Neuron> Layer;

class Neuron
{
public:
  Neuron(unsigned numOutputs, unsigned myIndex);
  void setOutputVal(double val) { m_outputVal = val; }
  double getOutputVal(void) const { return m_outputVal; }
  void feedForward(const Layer &prevLayer);
  void calcOutputGradients(double targetVal);
  void calcHiddenGradients(const Layer &nextLayer);
  void updateInputWeights(Layer &prevLayer);
private:
  static double eta;   // [0.0..1.0] overall net training rate
  static double alpha; // [0.0..n] multiplier of last weight change (momentum)
  static double transferFunction(double x);
  static double transferFunctionDerivative(double x);
  static double randomWeight(void) { return std::rand() / double(RAND_MAX); }
  double sumDOW(const Layer &nextLayer) const;
  double m_outputVal;
  std::vector<Connection> m_outputWeights;
  unsigned m_myIndex;
  double m_gradient;
};

#endif // NEURON_H
