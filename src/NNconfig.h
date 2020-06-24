#ifndef NNCONFIG_H
#define NNCONFIG_H

///< Momentum, multiplier of last deltaWeight, [0.0..1.0]
const double ALFA = 0.5;
///< Overall net learning rate, [0.0..1.0]
const double ETA = 0.15;
const int MAX_ITERATIONS = 100'000;
const double MIN_RECENT_AVERAGE_ERROR = 0.01;

#endif
