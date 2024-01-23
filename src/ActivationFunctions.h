#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H

#include "NNdef.h"

#include <cmath>
#include <functional>
#include <map>
#include <string>

namespace nn {

using action_functions_t =
   std::map<std::string,
            std::pair<nndef::action_function_t, nndef::action_function_t>>;

/// Action function: tanh - output range [-1.0..1.0].
inline double tanh(double z)
{
   return std::tanh(z);
}

/// Action function tanh derivative.
inline double tanh_derivative(double z)
{
   return 1.0 - std::tanh(z) * std::tanh(z);
}

/// Action function: sigmoid - output range [0.0..1.0].
inline double sigmoid(double z)
{
   return 1.0 / (1.0 + std::exp(-z));
}

/// Action function: sigmoid derivative.
inline double sigmoid_derivative(double z)
{
   return sigmoid(z) * (1 - sigmoid(z));
}

inline double relu(double z)
{
   return std::max(0.0, z);
}

inline double relu_derivative(double z)
{
   return z >= 0 ? 1 : 0;
}

inline double leaky_relu(double z)
{
   const double a{0.01};
   return z >= 0 ? z : a * z;
}

inline double leaky_relu_derivative(double z)
{
   const double a{0.01};
   return z >= 0 ? 1 : a;
}

inline action_functions_t act_fs{
   {"tanh", std::make_pair(nn::tanh, nn::tanh_derivative)},
   {"sigmoid", std::make_pair(nn::sigmoid, nn::sigmoid_derivative)},
   {"relu", std::make_pair(nn::relu, nn::relu_derivative)},
   {"leaky_relu", std::make_pair(nn::leaky_relu, nn::leaky_relu_derivative)}};

} // namespace nn

#endif