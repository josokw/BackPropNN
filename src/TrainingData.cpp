#include "TrainingData.h"

#include <iostream>
#include <string>

std::ostream &operator<<(std::ostream &os, const TrainingData &trnData)
{
   os << "topology: ";
   for (auto layerSize : trnData.topology_) {
      os << layerSize << ' ';
   }
   os << std::endl;
   for (auto &io : trnData.in_out_all_) {
      os << "in:";
      for (auto x : io.first) {
         os << " " << x;
      }
      os << std::endl;
      os << "out:";
      for (auto x : io.second) {
         os << " " << x;
      }
      os << std::endl;
   }

   os << std::endl;

   return os;
}

std::istream &operator>>(std::istream &is, TrainingData &trnData)
{
   trnData.topology_.clear();

   if (!is.eof()) {
      std::string line;
      getline(is, line);
      // Remove trailing white spaces
      line.erase(line.find_last_not_of(" \t") + 1);
      std::stringstream lineStream1{line};
      std::string label;
      lineStream1 >> label;

      if (label == "topology:") {
         while (!lineStream1.eof()) {
            unsigned n;
            lineStream1 >> n;
            trnData.topology_.push_back(n);
         }
      }
   }

   while (!is.eof()) {
      nndef::in_out_pair_t iop;

      std::string line;
      getline(is, line);
      // Remove trailing white spaces
      line.erase(line.find_last_not_of(" \t") + 1);
      std::stringstream lineStream2{line};
      std::string label;
      lineStream2 >> label;

      if (label == "in:") {
         while (!lineStream2.eof()) {
            double inputValue;
            while (lineStream2 >> inputValue) {
               iop.first.push_back(inputValue);
            }
         }
      }

      getline(is, line);
      // Remove trailing white spaces
      line.erase(line.find_last_not_of(" \t") + 1);
      std::stringstream lineStream3{line};
      lineStream3 << line;
      lineStream3 >> label;

      if (label == "out:") {
         while (!lineStream3.eof()) {
            double targetValue;
            while (lineStream3 >> targetValue) {
               iop.second.push_back(targetValue);
            }
         }
      }
      trnData.in_out_all_.push_back(iop);
   }

   return is;
}

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

nndef::in_out_pair_t TrainingData::getRandomChoosenInOut() const
{
   int randomIndex = random() % in_out_all_.size();
   return in_out_all_[randomIndex];
}
