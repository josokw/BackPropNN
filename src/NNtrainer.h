#ifndef NNTRAINER_H
#define NNTRAINER_H

#include "NNdef.h"

#include <cstddef>
#include <memory>

class Net;
class TrainingData;

/// Interface notified once per training pass with the current model state.
class NNtrainerObserver
{
public:
   virtual ~NNtrainerObserver() = default;

   /// Called once per training pass after feed-forward and result capture.
   virtual void onPass(std::size_t pass, bool show, Net &net,
                       const nndef::values_layer_t &inputVals,
                       const nndef::values_layer_t &resultVals,
                       const nndef::values_layer_t &targetVals) = 0;

   /// Called when the training loop has terminated.
   virtual void onFinished(Net &net) { (void)net; }
};

/// Class NNtrainer manages the training of a backprop NN.
class NNtrainer
{
public:
   NNtrainer(Net &net, TrainingData &traningData);
   ~NNtrainer() = default;

   void train();
   void setObserver(std::shared_ptr<NNtrainerObserver> observer);
   [[nodiscard]] bool hasObserver() const { return observer_ != nullptr; }

private:
   Net &net_;
   TrainingData &trainingData_;
   std::size_t trainingPass_{0UL};
   std::shared_ptr<NNtrainerObserver> observer_;
};

#endif
