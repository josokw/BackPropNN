// BPNN.cpp --------------------------------------------------------------------
// Refactored by Jos Onokiewicz
//
// David Miller, http://millermattson.com/dave
// See the associated video for instructions: http://vimeo.com/19569529

#include "Net.h"
#include "TrainingData.h"
#include <vector>
#include <iostream>
#include <cassert>

using namespace std;

void showVectorVals(const string& label, const vector<double>& v)
{
  cout << label << " ";
  for (const auto e: v) {
    cout << e << " ";
  }
  cout << endl;
}

int main()
{
  TrainingData trainData("trainingData.txt");
  // e.g., { 3, 2, 1 }
  vector<unsigned> topology;
  trainData.getTopology(topology);

  Net myNet(topology);
  vector<double> inputVals;
  vector<double> targetVals;
  vector<double> resultVals;
  int trainingPass = 0;

  while (!trainData.isEof()) {
    ++trainingPass;
    cout << endl << "Pass " << trainingPass;
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
    cout << "Net recent average error: "
         << myNet.getRecentAverageError() << endl;
  }
  cout << endl << "Done" << endl;
}
