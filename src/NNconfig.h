#ifndef NNCONFIG_H
#define NNCONFIG_H

///< Momentum, multiplier of last deltaWeight, [0.0..1.0]
inline double ALPHA = 0.5;
///< Overall net learning rate, [0.0..1.0]
inline double ETA = 0.15;

const unsigned long MAX_ITERATIONS = 500'000;
const double MIN_RECENT_AVERAGE_ERROR = 0.01;

#endif
