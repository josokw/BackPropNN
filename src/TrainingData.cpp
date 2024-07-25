#include "TrainingData.h"
#include "NNdef.h"
#include "Neuron.h"

#include <iostream>
#include <map>
#include <sstream>
#include <string>

std::ostream &operator<<(std::ostream &os, const TrainingData &trnData)
{
   os << "momentum (ALPHA): " << trnData.ALPHA << "\n";
   os << "learning rate (ETA): " << trnData.ETA << "\n";
   os << "topology: \n";
   for (int index = 0; auto layerSize : trnData.topology_) {
      os << "  " << layerSize << "  "
         << trnData.activation_function_names_[index++] << "\n";
   }
   os << "\n";

   for (auto &io : trnData.in_out_all_) {
      os << "in:";
      for (int line = 0; auto x : io.first) {
         if (trnData.show_max_inputs != 0 and
             (line++ % trnData.show_max_inputs) == 0) {
            os << "\n";
         }
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

   auto ignoreCommentWhiteSpace = [&is, &trnData]() {
      std::string line;
      while (line.empty() && !is.eof()) {
         getline(is, line);
         ++trnData.line_;
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

      if (trnData.semantic_actions_.find(label) !=
          trnData.semantic_actions_.end()) {
         trnData.semantic_actions_[label](lineStream1, trnData);
      } else {
         if (!label.empty()) {
            std::cerr << "=== SYNTAX ERROR line [" << trnData.line_ << "]: '"
                      << label << "' ???\n"
                      << std::endl;
            std::exit(EXIT_FAILURE);
         }
         break;
      }
   }

   return is;
}

TrainingData::TrainingData()
   : topology_{}
   , activation_function_names_{}
   , in_out_all_{}
   , semantic_actions_{}
{
   semantic_actions_["ALPHA:"] = sa_ALPHA;
   semantic_actions_["momentum:"] = sa_ALPHA;
   semantic_actions_["ETA:"] = sa_ETA;
   semantic_actions_["learning_rate:"] = sa_ETA;
   semantic_actions_["topology:"] = sa_topology;
   semantic_actions_["actionfs:"] = sa_activationfs;
   semantic_actions_["in:"] = sa_in;
   semantic_actions_["out:"] = sa_out;
   semantic_actions_["show_max_inputs:"] = sa_show_max_inputs;
   semantic_actions_["show_max_outputs:"] = sa_show_max_outputs;
   semantic_actions_["output_names:"] = sa_output_names;
}

nndef::in_out_pair_t TrainingData::getRandomChoosenInOut() const
{
   auto randomIndex = std::rand() % in_out_all_.size();
   return in_out_all_[randomIndex];
}

void sa_ALPHA(std::stringstream &lineStream, TrainingData &trainingData)
{
   while (not lineStream.eof() and not lineStream.fail()) {
      lineStream >> trainingData.ALPHA;
   }
   if (lineStream.fail() or
       not(trainingData.ALPHA > 0.0 and trainingData.ALPHA < 1.0)) {
      std::cerr << "=== ERROR line [" << trainingData.line_
                << "]: ALPHA (momentum) not in range (0, 1)\n\n";
      std::exit(EXIT_FAILURE);
   }
   Neuron::set_alpha(trainingData.ALPHA);
}

void sa_ETA(std::stringstream &lineStream, TrainingData &trainingData)
{
   while (not lineStream.eof() and not lineStream.fail()) {
      lineStream >> trainingData.ETA;
   }
   if (lineStream.fail() or
       not(trainingData.ETA > 0.0 and trainingData.ETA < 1.0)) {
      std::cerr << "=== ERROR line [" << trainingData.line_
                << "]: ETA (learning rate) not in range (0,1)\n\n";
      std::exit(EXIT_FAILURE);
   }
   Neuron::set_eta(trainingData.ETA);
}

void sa_topology(std::stringstream &lineStream, TrainingData &trainingData)
{
   while (not lineStream.eof() and not lineStream.fail()) {
      int n{0};
      lineStream >> n;
      if (lineStream.fail() or n <= 0) {
         std::cerr << "=== ERROR line [" << trainingData.line_
                   << "]: topology data not > 0\n\n";
         std::exit(EXIT_FAILURE);
      }
      trainingData.topology_.push_back(n);
   }
}

void sa_activationfs(std::stringstream &lineStream, TrainingData &trainingData)
{
   while (!lineStream.eof()) {
      std::string af_name;
      lineStream >> af_name;
      if (af_name != "inputs" and
          std::find(nndef::all_activation_function_names.begin(),
                    nndef::all_activation_function_names.end(),
                    af_name) == nndef::all_activation_function_names.end()) {
         std::cerr << "=== ERROR line [" << trainingData.line_
                   << "]: unknown activation function name '" << af_name
                   << "'\n\n";
         std::exit(EXIT_FAILURE);
      }

      trainingData.activation_function_names_.push_back(af_name);
   }
   if (trainingData.topology_.size() !=
       trainingData.activation_function_names_.size()) {
      std::cerr
         << "=== ERROR line [" << trainingData.line_
         << "]: number of activation function names not equal to topology\n\n";
      std::exit(EXIT_FAILURE);
   }
}

void sa_in(std::stringstream &lineStream, TrainingData &trainingData)
{
   nndef::in_out_pair_t iop;
   while (!lineStream.eof()) {
      double inputValue{0.0};
      while (lineStream >> inputValue) {
         iop.first.push_back(inputValue);
      }
   }
   trainingData.in_out_all_.push_back(iop);
}

void sa_out(std::stringstream &lineStream, TrainingData &trainingData)
{
   nndef::in_out_pair_t &iop =
      trainingData.in_out_all_[trainingData.in_out_all_.size() - 1];
   while (!lineStream.eof()) {
      double targetValue;
      while (lineStream >> targetValue) {
         iop.second.push_back(targetValue);
      }
   }
}

void sa_show_max_inputs(std::stringstream &lineStream,
                        TrainingData &trainingData)
{
   while (not lineStream.eof() and not lineStream.fail()) {
      lineStream >> trainingData.show_max_inputs;
      if (lineStream.fail() or trainingData.show_max_inputs < 0) {
         std::cerr << "=== ERROR line [" << trainingData.line_
                   << "]: show_max_inputs is not >= 0\n\n";
         std::exit(EXIT_FAILURE);
      }
   }
}

void sa_show_max_outputs(std::stringstream &lineStream,
                         TrainingData &trainingData)
{
   while (not lineStream.eof() and not lineStream.fail()) {
      lineStream >> trainingData.show_max_outputs;
      if (lineStream.fail() or trainingData.show_max_outputs < 0) {
         std::cerr << "=== ERROR line [" << trainingData.line_
                   << "]: show_max_outputs is not >= 0\n\n";
         std::exit(EXIT_FAILURE);
      }
   }
}

void sa_output_names(std::stringstream &lineStream, TrainingData &trainingData)
{
   trainingData.output_names.clear();
   while (!lineStream.eof()) {
      std::string output_name;
      while (lineStream >> output_name) {
         trainingData.output_names.push_back(output_name);
      }
   }

   auto str_compare = [](const std::string &a, const std::string &b) {
      return (a.size() < b.size());
   };

   trainingData.max_size_ =
      std::string(*std::max_element(trainingData.output_names.begin(),
                                    trainingData.output_names.end(),
                                    str_compare))
         .size();
}
