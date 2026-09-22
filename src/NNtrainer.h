#ifndef NNTRAINER_H
#define NNTRAINER_H

#include "NNdef.h"

#include <atomic>
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
   virtual void onFinished(Net &net, double elapsedMs = 0)
   {
      (void)net;
      (void)elapsedMs;
   }
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

    /// Asks the training loop to stop at the next sample boundary.
    void requestStop() noexcept { cancel_.store(true, std::memory_order_relaxed); }
    /// Sets or clears the cooperative pause state of the training loop.
    void setPaused(bool paused) noexcept
    {
       paused_.store(paused, std::memory_order_relaxed);
    }
    [[nodiscard]] bool stopRequested() const noexcept
    {
       return cancel_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] bool isPaused() const noexcept
    {
       return paused_.load(std::memory_order_relaxed);
    }

private:
    Net &net_;
    TrainingData &trainingData_;
    std::size_t trainingPass_{0UL};
    std::shared_ptr<NNtrainerObserver> observer_;
    std::atomic<bool> cancel_{false};
    std::atomic<bool> paused_{false};
};

#endif
