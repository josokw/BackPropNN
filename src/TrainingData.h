#ifndef TRAININGDATA_H
#define TRAININGDATA_H

#include "NNdef.h"

#include <fstream>

/// Class TrainingData for managing training data.
class TrainingData
{
   friend std::ostream &operator<<(std::ostream &os, const TrainingData &trnData);
   friend std::istream &operator>>(std::istream &is, TrainingData &trnData);
   
public:
   TrainingData();
   ~TrainingData() = default;

   const nndef::topology_t &getTopology() const { return topology_; }
   nndef::in_out_pair_t getRandomChoosenInOut() const;

private:
   nndef::topology_t topology_;
   nndef::in_out_all_pairs_t in_out_all_;
};

#endif // TRAININGDATA_H
