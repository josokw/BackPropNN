// Refactored and extended by Jos Onokiewicz
//
// David Miller, http://millermattson.com/dave
// See the associated video for instructions: http://vimeo.com/19569529

#include "AppInfo.h"
#include "NNtrainer.h"
#include "Net.h"
#include "OSstate.h"
#include "TrainingData.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void showInOut(const std::string &message, const TrainingData &trainingData,
               Net &net, int precision = 3);

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

   try {
      TrainingData trainingData;

      trainingDataStream >> trainingData;
      std::cout << trainingData;

      Net myNet{trainingData.getTopology(),
                trainingData.getActionFunctionNames()};
      NNtrainer nntr{myNet, trainingData};

      nntr.train();

      showInOut("\n- Results after training:", trainingData, myNet);
   }
   catch (std::exception &e) {
      std::cerr << "ERROR: " << e.what() << "\n";
   }
   catch (...) {
      std::cerr << "ERROR: unknown exception\n";
   }

   std::cout << "\n*** " APPNAME_VERSION " ready\n\n";

   return 0;
}

void showInOut(const std::string &message, const TrainingData &trainingData,
               Net &net, int precision)
{
   OSstate state(std::cout);

   std::cout << message;

   auto in_out{trainingData.getInOut()};

   std::cout << std::showpos << std::setw(6) << std::fixed
             << std::setprecision(precision) << '\n';

   for (auto io : in_out) {
      for (auto i : io.first) {
         std::cout << i << " ";
      }
      std::cout << " ==> ";
      net.feedForward(io.first);
      nndef::values_layer_t output;
      net.getResults(output);
      for (auto o : output) {
         std::cout << o << " ";
      }
      std::cout << std::endl;
   }
}
