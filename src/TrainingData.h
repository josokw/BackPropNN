#ifndef TRAININGDATA_H
#define TRAININGDATA_H

#include "NNconfig.h"
#include "NNdef.h"

#include <fstream>
#include <sstream>

/// Class TrainingData for managing training data.
class TrainingData
{
   friend std::ostream &operator<<(std::ostream &os,
                                   const TrainingData &trnData);
   friend std::istream &operator>>(std::istream &is, TrainingData &trnData);

public:
   TrainingData();
   ~TrainingData() = default;

   [[nodiscard]] const nndef::topology_t &getTopology() const
   {
      return topology_;
   }
   [[nodiscard]] const nndef::activation_function_names_t &
   getActionFunctionNames() const
   {
      return activation_function_names_;
   }
   [[nodiscard]] nndef::in_out_pair_t getRandomChoosenInOut() const;
   [[nodiscard]] nndef::in_out_all_pairs_t getInOut() const
   {
      return in_out_all_;
   }

   ///< Momentum, multiplier of last deltaWeight, [0.0..1.0]
   double ALPHA{DEFAULT_ALPHA};
   ///< Overall net learning rate, [0.0..1.0]
   double ETA{DEFAULT_ETA};
   ///< Max number of input values showed in a line (2D view)
   int show_max_inputs{0};
   ///< Max number of output values showed in a line (2D view)
   int show_max_outputs{0};
   ///< Seed for the Mersenne Twister RNG, [reproducibility]
   std::size_t seed{DEFAULT_SEED};
   ///< Names for output values
   std::vector<std::string> output_names{};

   // private:
   nndef::topology_t topology_;
   nndef::activation_function_names_t activation_function_names_;
   nndef::in_out_all_pairs_t in_out_all_;
   nndef::semantic_activations_t semantic_actions_;
   size_t max_size_{0};
   /// Line number in config file
   size_t line_{0};
};

/// Semantic actions for labels in training data
/// @todo add more checking correct syntax
void sa_ALPHA(std::stringstream &lineStream, TrainingData &trainingData);
void sa_ETA(std::stringstream &lineStream, TrainingData &trainingData);
void sa_topology(std::stringstream &lineStream, TrainingData &trainingData);
void sa_activationfs(std::stringstream &lineStream, TrainingData &trainingData);
void sa_in(std::stringstream &lineStream, TrainingData &trainingData);
void sa_out(std::stringstream &lineStream, TrainingData &trainingData);
void sa_show_max_inputs(std::stringstream &lineStream,
                        TrainingData &trainingData);
void sa_show_max_outputs(std::stringstream &lineStream,
                         TrainingData &trainingData);
void sa_seed(std::stringstream &lineStream, TrainingData &trainingData);
void sa_output_names(std::stringstream &lineStream, TrainingData &trainingData);

#endif // TRAININGDATA_H
