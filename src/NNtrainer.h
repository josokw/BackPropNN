#ifndef NNTRAINER_H
#define NNTRAINER_H

#include <cstddef>

class Net;
class TrainingData;

/// Class NNtrainer manages the training of a backprop NN.
class NNtrainer
{
public:
   NNtrainer(Net &net, TrainingData &traningData);
   ~NNtrainer() = default;

   void train();

private:
   Net &net_;
   TrainingData &trainingData_;
   std::size_t trainingPass_{0UL};
};

#endif
