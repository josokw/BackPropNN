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

private:
   using in_out_pair_t = std::pair<std::vector<double>, std::vector<double>>;
   using in_out_all_t = std::vector<in_out_pair_t>;
   
   std::ifstream trainingDataFile_;
   std::vector<unsigned> topology_;
   in_out_all_t in_out_all_;
};

#endif // TRAININGDATA_H
