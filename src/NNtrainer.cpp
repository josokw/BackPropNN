#include "NNtrainer.h"
#include "NNdef.h"
#include "Net.h"
#include "TrainingData.h"

#include <cassert>
#include <iostream>
#include <vector>

NNtrainer::NNtrainer(Net &net, TrainingData &trainingData)
   : net_{net}
   , trainingData_{trainingData}
{
}

void NNtrainer::train()
{
   auto showVectorVals = [](const std::string &label,
                            const std::vector<double> &v) {
      std::cout << label << " ";
      for (const auto e : v) {
         std::cout << e << " ";
      }
      std::cout << std::endl;
   };

   while (trainingPass_ < 10000 && net_.getRecentAverageError() > 0.01) {
      ++trainingPass_;

      nndef::values_layer_t resultVals;
      const auto [inputVals, targetVals] = trainingData_.getRandomChoosenInOut();

      std::cout << "\n-- Pass " << trainingPass_;
      showVectorVals(": Inputs:", inputVals);
      net_.feedForward(inputVals);
      // Collect the net's actual output results:
      net_.getResults(resultVals);
      showVectorVals("Outputs:", resultVals);
      // Train the net what the outputs should have been:
      showVectorVals("Targets:", targetVals);
      assert(targetVals.size() == net_.topology().back());
      net_.backProp(targetVals);

      // Report how well the training is working, average over recent
      // samples:
      std::cout << "Net recent average error: " << net_.getRecentAverageError()
                << std::endl;
   }
}
