#include "TrainingData.h"

#include <iostream>
#include <string>

TrainingData::TrainingData(const std::string &filename)
   : trainingDataFile_{filename.c_str()}
   , topology_{}
{
   if (not trainingDataFile_) {
   }
}

TrainingData::~TrainingData()
{
   trainingDataFile_.close();
}

const std::vector<unsigned> &TrainingData::setTopology()
{
   std::string line;
   getline(trainingDataFile_, line);
   std::stringstream ss{line};
   std::string label;
   ss >> label;

   if (isEof() || label != "topology:") {
      abort();
   }
   while (!ss.eof()) {
      unsigned n;
      ss >> n;
      topology_.push_back(n);
   }
   return topology_;
}

unsigned TrainingData::getNextInputs(std::vector<double> &inputVals)
{
   inputVals.clear();

   std::string line;
   getline(trainingDataFile_, line);
   std::stringstream ss{line};
   std::string label;
   ss >> label;

   if (label == "in:") {
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
   std::stringstream ss{line};
   std::string label;
   ss >> label;
   
   if (label == "out:") {
      double oneValue;
      while (ss >> oneValue) {
         targetOutputVals.push_back(oneValue);
      }
   }
   return targetOutputVals.size();
}
