#ifndef TUI_H
#define TUI_H

#include "NNtrainer.h"

#include <cstddef>
#include <string>

class Net;
class TrainingData;

namespace tui {

/// Live training dashboard rendered with FTXUI (a Python-Rich-style TUI).
/// When compiled without BPNN_TUI the dashboard degrades to a no-op and the
/// classic text output is used instead.
class Dashboard final : public NNtrainerObserver
{
public:
   Dashboard(Net &net, const TrainingData &trnData,
             std::string inputFileName = {},
             std::size_t renderEvery = 50);
   ~Dashboard() override = default;

   /// Whether a terminal UI should be shown (tty and BPNN_TUI env override).
   [[nodiscard]] static bool wantsTui() noexcept;

   void onPass(std::size_t pass, bool show, Net &net,
               const nndef::values_layer_t &inputVals,
               const nndef::values_layer_t &resultVals,
               const nndef::values_layer_t &targetVals) override;
   void onFinished(Net &net) override;

private:
   void render(Net &net);

   Net &net_;
   const TrainingData &trnData_;
   std::string inputFileName_;
   std::size_t renderEvery_;
   std::size_t lastPass_{0};
   bool finished_{false};
   nndef::values_layer_t lastInputs_;
   nndef::values_layer_t lastResults_;
   nndef::values_layer_t lastTargets_;
   std::string resetPosition_;
};

} // namespace tui

#endif // TUI_H