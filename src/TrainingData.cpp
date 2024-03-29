#include "TrainingData.h"
#include "NNconfig.h"

#include <iostream>
#include <sstream>
#include <string>

std::ostream &operator<<(std::ostream &os, const TrainingData &trnData)
{
   os << "momentum (ALPHA): " << ALPHA << "\n";
   os << "learning rate (ETA): " << ETA << "\n";
   os << "topology: \n";
   for (int index = 0; auto layerSize : trnData.topology_) {
      os << "  " << layerSize << "  " << trnData.action_function_names_[index++]
         << "\n";
   }
   os << "\n";

   for (auto &io : trnData.in_out_all_) {
      os << "in:";
      for (auto x : io.first) {
         os << " " << x;
      }
      os << "\nout:";
      for (auto x : io.second) {
         os << " " << x;
      }
      os << "\n";
   }
   os << std::endl;

   return os;
}

std::istream &operator>>(std::istream &is, TrainingData &trnData)
{
   trnData.topology_.clear();

   auto ignoreCommentWhiteSpace = [&is]() {
      std::string line;
      while (line.empty() && !is.eof()) {
         getline(is, line);
         // Remove single line comment after #
         if (auto pos = line.find('#'); pos != std::string::npos) {
            line.erase(pos);
         }
         // Remove trailing white spaces
         line.erase(line.find_last_not_of(" \t") + 1);
      }
      return line;
   };

   while (!is.eof()) {
      std::stringstream lineStream1{ignoreCommentWhiteSpace()};

      std::string label;
      lineStream1 >> label;

      if (label == "momentum:" or label == "ALPHA:") {
         while (!lineStream1.eof()) {
            lineStream1 >> ALPHA;
         }
      }

      if (label == "learning_rate:" or label == "ETA") {
         while (!lineStream1.eof()) {
            lineStream1 >> ETA;
         }
      }

      if (label == "topology:") {
         while (!lineStream1.eof()) {
            unsigned n;
            lineStream1 >> n;
            trnData.topology_.push_back(n);
         }
      }
      if (label == "actionfs:") {
         while (!lineStream1.eof()) {
            std::string af_name;
            lineStream1 >> af_name;
            trnData.action_function_names_.push_back(af_name);
         }
         break;
      }
   }

   while (!is.eof()) {
      nndef::in_out_pair_t iop;
      std::stringstream lineStream2{ignoreCommentWhiteSpace()};
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

      std::stringstream lineStream3{ignoreCommentWhiteSpace()};
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
   , action_function_names_{}
   , in_out_all_{}
{
}

nndef::in_out_pair_t TrainingData::getRandomChoosenInOut() const
{
   auto randomIndex = std::rand() % in_out_all_.size();
   return in_out_all_[randomIndex];
}
