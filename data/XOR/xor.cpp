#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

int main(int argc, char* argv[])
{
  int nTrainingInputs {0};
  int nHiddenNeurons {0};
  switch (argc)
  {
    case 1:
      nTrainingInputs = 100;
      nHiddenNeurons = 4;
      break;
    case 2:
      nTrainingInputs = stoi(argv[1]);
      nHiddenNeurons = 4;
      break;
    case 3:
      nTrainingInputs = stoi(argv[1]);
      nHiddenNeurons = stoi(argv[2]);
      break;
    default:
      cerr << "Usage: " << argv[0] << " <number of training sets>\n";
      cerr << "       " << argv[0] << " <number of training sets>"
              " <number of hidden neurons>\n";
      break;
  }
  // Random training sets for XOR: 2 inputs and 1 output
  cout << "topology: 2 " << nHiddenNeurons << " 1\n";
  for (int i = 0; i < nTrainingInputs; ++i)
  {
    int in1 {int(2.0 * rand() / RAND_MAX)};
    int in2 {int(2.0 * rand() / RAND_MAX)};
    int out {in1 ^ in2};
    cout << "in: " << in1 << ".0 " << in2 << ".0" << endl;
    cout << "out: " << out << ".0" << endl;
  }

  return 0;
}
