#ifndef TRAININGDATA_H
#define TRAININGDATA_H

#include "NNdef.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

/// Class to read training data from a text file.
class TrainingData
{
   friend std::ostream &operator<<(std::ostream &os, const TrainingData &trnData);
   friend std::istream &operator>>(std::istream &is, TrainingData &trnData);
   
public:
   TrainingData(const std::string &filename);
   ~TrainingData();

   bool isEof() const { return trainingDataFile_.eof(); }
   const std::vector<unsigned> &setTopology();
   // Returns the number of input values read from the file:
   unsigned getNextInputs(std::vector<double> &inputVals);
   unsigned getTargetOutputs(std::vector<double> &targetOutputVals);

   nndef::in_out_pair_t getRandomChoosenInOut() const;

private:
   std::ifstream trainingDataFile_;
   nndef::topology_t topology_;
   nndef::in_out_all_pairs_t in_out_all_;
};

#endif // TRAININGDATA_H
