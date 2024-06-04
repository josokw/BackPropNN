#include "NNtrainer.h"
#include "NNconfig.h"
#include "NNdef.h"
#include "Net.h"
#include "TrainingData.h"

#include <cassert>
#include <iomanip>
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
         std::cout << std::setw(6) << std::fixed << std::setprecision(3) << e
                   << " ";
      }
      std::cout << std::endl;
   };

   while (trainingPass_ < MAX_ITERATIONS &&
          net_.getRecentAverageError() > MIN_RECENT_AVERAGE_ERROR) {
      ++trainingPass_;

      nndef::values_layer_t resultVals;
      const auto [inputVals, targetVals] =
         trainingData_.getRandomChoosenInOut();

      if (do_show(trainingPass_, net_.getRecentAverageError())) {
         std::cout << "\n-- Pass " << trainingPass_;
         showVectorVals("\nInputs: ", inputVals);
         std::cout << net_;
      }

      net_.feedForward(inputVals);
      // Collect the net's actual output results:
      net_.getResults(resultVals);

      if (do_show(trainingPass_, net_.getRecentAverageError())) {
         showVectorVals("Outputs:", resultVals);
         // Train the net what the outputs should have been:
         showVectorVals("Targets:", targetVals);
         assert(targetVals.size() == net_.topology().back());
      }

      /// @todo Check call to net_.backProp(targetVals);
      // net_.backProp(targetVals);

      if (do_show(trainingPass_, net_.getRecentAverageError())) {
         // Report how well the training is working, average over recent
         // samples:
         // std::cout << "Net recent average error: "
         //           << net_.getRecentAverageError() << std::endl;
      }

      net_.backProp(targetVals);
      // std::cout << net_;
   }
}
