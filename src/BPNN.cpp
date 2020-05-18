// BPNN.cpp --------------------------------------------------------------------
// Refactored by Jos Onokiewicz
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

   std::cout << "*** " APPNAME_VERSION " started\n\n";

   std::ifstream trainingDataStream{argv[1]};
   if (not trainingDataStream) {
      std::cerr << "ERROR: file " << argv[1] << " can not be opened\n";
   }

   TrainingData trainData{argv[1]};
   auto topology{trainData.setTopology()};
   Net myNet{topology};

   trainingDataStream >> trainData;
   std::cout << trainData;

   NNtrainer nntr{myNet, trainData};

   nntr.train();

   std::cout << "\n*** " APPNAME_VERSION " ready\n\n";

   return 0;
}
