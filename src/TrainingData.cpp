#include "TrainingData.h"

#include <iostream>
#include <sstream>
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

   auto ignoreWhiteSpace = [&is]() {
      std::string line;
      while (line.empty() && !is.eof()) {
         getline(is, line);
         // Remove trailing white spaces
         line.erase(line.find_last_not_of(" \t") + 1);
      }
      return line;
   };

   if (!is.eof()) {
      std::stringstream lineStream1{ignoreWhiteSpace()};

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
      std::stringstream lineStream2{ignoreWhiteSpace()};
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

      std::stringstream lineStream3{ignoreWhiteSpace()};
      lineStream3 >> label;

      if (label == "out:") {
         while (!lineStream3.eof()) {
            double targetValue;
            while (lineStream3 >> targetValue) {
               iop.second.push_back(targetValue);
            }
         }
          trnData.in_out_all_.push_back(iop);
      }
   }

   return is;
}

TrainingData::TrainingData()
   : topology_{}
   , in_out_all_{}
{
}

nndef::in_out_pair_t TrainingData::getRandomChoosenInOut() const
{
   auto randomIndex = random() % in_out_all_.size();
   return in_out_all_[randomIndex];
}
