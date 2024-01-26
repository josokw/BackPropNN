// Refactored and extended by Jos Onokiewicz
//
// David Miller, http://millermattson.com/dave
// See the associated video for instructions: http://vimeo.com/19569529

#include "AppInfo.h"
#include "NNtrainer.h"
#include "Net.h"
#include "TrainingData.h"

#include <iostream>
#include <sstream>
#include <vector>

/// Back Propagating Neural Network
/// \warning No checking of command line parameters (not robust)
int main(int argc, char *argv[])
{
   if (argc != 2) {
      std::cerr << "Usage " << argv[0] << " <file name>\n\n";
      exit(1);
   }

   std::cout << "*** " APPNAME_VERSION " started\n";

   std::ifstream trainingDataStream{argv[1]};
   if (not trainingDataStream) {
      std::cerr << "ERROR: file " << argv[1] << " can not be opened\n";
      exit(EXIT_FAILURE);
   }

   std::cout << "*** config file: " << argv[1] << "\n\n";

   TrainingData trainData;

   trainingDataStream >> trainData;
   std::cout << trainData;

   Net myNet{trainData.getTopology(), trainData.getActionFunctionNames()};
   NNtrainer nntr{myNet, trainData};

   nntr.train();

   std::cout << "\n*** " APPNAME_VERSION " ready\n\n";

   return 0;
}
