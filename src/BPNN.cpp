// BPNN.cpp --------------------------------------------------------------------
// Refactored by Jos Onokiewicz
//
// David Miller, http://millermattson.com/dave
// See the associated video for instructions: http://vimeo.com/19569529

#include "Net.h"
#include "TrainingData.h"
#include "NNtrainer.h"

#include <cassert>
#include <iostream>
#include <vector>

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

   NNtrainer nntr{myNet, trainData};
  
   nntr.train();

   return 0;
}
