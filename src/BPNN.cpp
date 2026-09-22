// Refactored and extended by Jos Onokiewicz
//
// David Miller, http://millermattson.com/dave
// See the associated video for instructions: http://vimeo.com/19569529

#include "AppInfo.h"
#include "NNtrainer.h"
#include "Net.h"
#include "OSstate.h"
#include "TrainingData.h"
#include "Tui.h"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
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
      return EXIT_FAILURE;
   }

   std::cout << "*** " APPNAME_VERSION " started\n";

   std::ifstream trainingDataStream{argv[1]};
   if (not trainingDataStream) {
      std::cerr << "ERROR: file " << argv[1] << " can not be opened\n";
      return EXIT_FAILURE;
   }

   std::cout << "*** config file: " << argv[1] << "\n\n";

   bool tuiActive = false;

   try {
      TrainingData trainingData;

      trainingDataStream >> trainingData;
      std::cout << trainingData;

      Net myNet{trainingData.getTopology(),
                trainingData.getActionFunctionNames()};
      NNtrainer nntr{myNet, trainingData};

#ifdef BPNN_TUI
      if (tui::Dashboard::wantsTui()) {
         auto dashboard =
            std::make_shared<tui::Dashboard>(myNet, trainingData, argv[1]);
         nntr.setObserver(dashboard);
         tuiActive = true;
         dashboard->run(nntr);
      }
#endif

      if (not tuiActive) {
         nntr.train();
      }

      if (not nntr.hasObserver()) {
         showInOut("\n- Results after training:", trainingData, myNet);
      }
   }
   catch (std::exception &e) {
      std::cerr << "ERROR: " << e.what() << "\n";
      return EXIT_FAILURE;
   }
   catch (...) {
      std::cerr << "ERROR: unknown exception\n";
      return EXIT_FAILURE;
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
      for (int index = 0; auto i : io.first) {
         if (trainingData.show_max_inputs != 0 and
             index++ % trainingData.show_max_inputs == 0) {
            std::cout << '\n';
         }
         std::cout << i << " ";
      }
      std::cout << "\n"
                << std::string(std::max(size_t(3), trainingData.max_size_), '=')
                << "> ";
      net.feedForward(io.first);
      nndef::values_layer_t output;
      net.getResults(output);
      for (auto index = 0UL; auto o : output) {
         if (trainingData.show_max_outputs != 0 and
             index % trainingData.show_max_outputs == 0) {
               if (index < trainingData.output_names.size()) {
            std::cout << '\n'
                      << std::setw(trainingData.max_size_)
                      << trainingData.output_names[index] << "  ";
               }
         }
         ++index;
         std::cout << o << " ";
      }
      std::cout << std::endl;
   }
}
