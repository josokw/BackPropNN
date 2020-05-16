#include "NNtrainer.h"
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
   Net::layer_t inputVals;
   Net::layer_t targetVals;
   Net::layer_t resultVals;

   auto showVectorVals = [](const std::string &label,
                            const std::vector<double> &v) {
      std::cout << label << " ";
      for (const auto e : v) {
         std::cout << e << " ";
      }
      std::cout << std::endl;
   };

   while (!trainingData_.isEof()) {
      ++trainingPass_;
      std::cout << "\nPass " << trainingPass_;
      // Get new input data and feed it forward:
      if (trainingData_.getNextInputs(inputVals) != net_.topology()[0]) {
         break;
      }
      showVectorVals(": Inputs:", inputVals);
      net_.feedForward(inputVals);
      // Collect the net's actual output results:
      net_.getResults(resultVals);
      showVectorVals("Outputs:", resultVals);
      // Train the net what the outputs should have been:
      trainingData_.getTargetOutputs(targetVals);
      showVectorVals("Targets:", targetVals);
      assert(targetVals.size() == net_.topology().back());
      net_.backProp(targetVals);

      // Report how well the training is working, average over recent samples:
      std::cout << "Net recent average error: " << net_.getRecentAverageError()
                << std::endl;
   }
   std::cout << "\nDone\n";
}
