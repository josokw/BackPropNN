#ifndef TRAININGDATA_H
#define TRAININGDATA_H

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

/// Silly class to read training data from a text file -- Replace This.
/// Replace class TrainingData with whatever you need to get input data into the
/// program, e.g., connect to a database, or take a stream of data from stdin,
/// or from a file specified by a command line argument, etc.
class TrainingData
{
public:
   TrainingData(const std::string &filename);
   ~TrainingData();

   bool isEof() { return trainingDataFile_.eof(); }
   const std::vector<unsigned> &setTopology();
   // Returns the number of input values read from the file:
   unsigned getNextInputs(std::vector<double> &inputVals);
   unsigned getTargetOutputs(std::vector<double> &targetOutputVals);

private:
   std::ifstream trainingDataFile_;
   std::vector<unsigned> topology_;
};

#endif // TRAININGDATA_H
