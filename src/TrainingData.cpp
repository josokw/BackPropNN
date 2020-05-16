#include "TrainingData.h"

#include <iostream>
#include <string>

TrainingData::TrainingData(const std::string &filename)
   : trainingDataFile_{filename.c_str()}
{
   if (not trainingDataFile_) {
   }
}

TrainingData::~TrainingData()
{
   trainingDataFile_.close();
}

void TrainingData::getTopology(std::vector<unsigned> &topology)
{
   std::string line("****");
   std::string label;

   getline(trainingDataFile_, line);
   std::stringstream ss(line);
   ss >> label;
   if (this->isEof() || label.compare("topology:") != 0) {
      abort();
   }
   while (!ss.eof()) {
      unsigned n;
      ss >> n;
      topology.push_back(n);
   }
}

unsigned TrainingData::getNextInputs(std::vector<double> &inputVals)
{
   inputVals.clear();

   std::string line;
   getline(trainingDataFile_, line);
   std::stringstream ss(line);

   std::string label;
   ss >> label;
   if (label.compare("in:") == 0) {
      double oneValue;
      while (ss >> oneValue) {
         inputVals.push_back(oneValue);
      }
   }
   return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(std::vector<double> &targetOutputVals)
{
   targetOutputVals.clear();

   std::string line;
   getline(trainingDataFile_, line);
   std::stringstream ss(line);

   std::string label;
   ss >> label;
   if (label.compare("out:") == 0) {
      double oneValue;
      while (ss >> oneValue) {
         targetOutputVals.push_back(oneValue);
      }
   }
   return targetOutputVals.size();
}
