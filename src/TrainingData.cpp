#include "TrainingData.h"
#include <iostream>

using namespace std;

TrainingData::TrainingData(const std::string& filename)
{
  trainingDataFile_.open(filename.c_str());
}

TrainingData::~TrainingData()
{
  trainingDataFile_.close();
}

void TrainingData::getTopology(vector<unsigned>& topology)
{
  string line("****");
  string label;

  getline(trainingDataFile_, line);
  stringstream ss(line);
  ss >> label;
  if (this->isEof() || label.compare("topology:") != 0) {
    abort();
  }
  while (!ss.eof()) {
    unsigned n;
    ss >> n;
    topology.push_back(n);
  }
}

unsigned TrainingData::getNextInputs(std::vector<double> &inputVals)
{
  inputVals.clear();

  string line;
  getline(trainingDataFile_, line);
  stringstream ss(line);

  string label;
  ss >> label;
  if (label.compare("in:") == 0) {
    double oneValue;
    while (ss >> oneValue) {
      inputVals.push_back(oneValue);
    }
  }
  return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(std::vector<double>& targetOutputVals)
{
  targetOutputVals.clear();

  string line;
  getline(trainingDataFile_, line);
  stringstream ss(line);

  string label;
  ss >> label;
  if (label.compare("out:") == 0) {
    double oneValue;
    while (ss >> oneValue) {
      targetOutputVals.push_back(oneValue);
    }
  }
  return targetOutputVals.size();
}
