#ifndef NNCONFIG_H
#define NNCONFIG_H

const unsigned long MAX_ITERATIONS = 1'000'000;
const double MIN_RECENT_AVERAGE_ERROR = 0.03;

inline bool do_show(unsigned long iteration, double average_error)
{
   return (iteration % 1000 == 0) or
          (average_error < 1.04 * MIN_RECENT_AVERAGE_ERROR);
}

#endif
