// BPNN.cpp --------------------------------------------------------------------
// Refactored by Jos Onokiewicz
//
// David Miller, http://millermattson.com/dave
// See the associated video for instructions: http://vimeo.com/19569529

#include "Net.h"
#include "TrainingData.h"

#include <cassert>
#include <iostream>
#include <vector>

void showVectorVals(const std::string &label, const std::vector<double> &v)
{
   std::cout << label << " ";
   for (const auto e : v) {
      std::cout << e << " ";
   }
   std::cout << std::endl;
}

/// Back Propagating Neural Network
/// \info No checking of command line parameters.
int main(int argc, char *argv[])
{
   if (argc != 2) {
      std::cerr << "Usage " << argv[0] << " <file name>\n\n";
      exit(1);
   }
   
   TrainingData trainData{argv[1]};
   auto topology{trainData.setTopology()};
   Net myNet{topology};
   std::vector<double> inputVals;
   std::vector<double> targetVals;
   std::vector<double> resultVals;
   int trainingPass = 0;

   while (!trainData.isEof()) {
      ++trainingPass;
      std::cout << "\nPass " << trainingPass;
      // Get new input data and feed it forward:
      if (trainData.getNextInputs(inputVals) != topology[0]) {
         break;
      }
      showVectorVals(": Inputs:", inputVals);
      myNet.feedForward(inputVals);
      // Collect the net's actual output results:
      myNet.getResults(resultVals);
      showVectorVals("Outputs:", resultVals);
      // Train the net what the outputs should have been:
      trainData.getTargetOutputs(targetVals);
      showVectorVals("Targets:", targetVals);
      assert(targetVals.size() == topology.back());
      myNet.backProp(targetVals);
      // Report how well the training is working, average over recent samples:
      std::cout << "Net recent average error: " << myNet.getRecentAverageError()
                << std::endl;
   }
   std::cout << "\nDone\n";
}
