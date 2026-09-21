#include "NNtrainer.h"
#include "NNconfig.h"
#include "NNdef.h"
#include "Net.h"
#include "TrainingData.h"

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

NNtrainer::NNtrainer(Net &net, TrainingData &trainingData)
   : net_{net}
   , trainingData_{trainingData}
   , observer_{nullptr}
{
}

void NNtrainer::setObserver(std::shared_ptr<NNtrainerObserver> observer)
{
   observer_ = std::move(observer);
}

void NNtrainer::train()
{
   auto showVectorVals = [](const std::string &label,
                            const std::vector<double> &v) {
      std::cout << label << " ";
      for (const auto e : v) {
         std::cout << std::setw(6) << std::fixed << std::setprecision(3) << e
                   << " ";
      }
      std::cout << std::endl;
   };

   const auto t_start = std::chrono::high_resolution_clock::now();

   while (trainingPass_ < MAX_ITERATIONS &&
          net_.getRecentAverageError() > MIN_RECENT_AVERAGE_ERROR) {
      ++trainingPass_;

      nndef::values_layer_t resultVals;
      const auto [inputVals, targetVals] =
         trainingData_.getRandomChoosenInOut();

      const bool show = do_show(trainingPass_, net_.getRecentAverageError());

      if (not observer_ and show) {
         std::cout << "\n-- Pass " << trainingPass_;
         showVectorVals("\nInputs: ", inputVals);
         std::cout << net_;
      }

      net_.feedForward(inputVals);
      net_.getResults(resultVals);

      assert(targetVals.size() == net_.topology().back());

      if (observer_) {
         observer_->onPass(trainingPass_, show, net_, inputVals, resultVals,
                           targetVals);
      } else if (show) {
         showVectorVals("Outputs:", resultVals);
         showVectorVals("Targets:", targetVals);
      }

      net_.backProp(targetVals);
   }

   if (observer_) {
      const auto t_ready = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double, std::milli> ms = t_ready - t_start;
      observer_->onFinished(net_, ms.count());
      return;
   }

   const auto t_ready = std::chrono::high_resolution_clock::now();
   const std::chrono::duration<double, std::milli> ms = t_ready - t_start;
   std::cout << std::fixed << std::setprecision(1) << "\n- Training took "
             << ms.count() << " ms\n";
}
