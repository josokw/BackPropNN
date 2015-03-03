#include <array>
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
  // Random training sets for counting inputs valued 1.0 ==> 
  // 0.1 (==1), 0.2 (==2), 0.3 (==3)
  cout << "topology: 3 " << nHiddenNeurons << " 1\n";
  array<int, 3> in;
  for (int i = 0; i < nTrainingInputs; ++i)
  {
    in[0] = (rand() / double(RAND_MAX)) > 0.5 ? 1: 0;
    in[1] = (rand() / double(RAND_MAX)) > 0.5 ? 1: 0;
    in[2] = (rand() / double(RAND_MAX)) > 0.5 ? 1: 0;
    double out { (in[0] + in[1] + in[2]) / 10.0};
    cout << "in: " << in[0] << ".0 " << in[1] << ".0 " << in[2] << ".0" << endl;
    cout << "out: " << out << endl;
  }

  return 0;
}
