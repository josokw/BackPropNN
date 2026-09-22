#ifndef TUI_H
#define TUI_H

#include "NNtrainer.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

class Net;
class TrainingData;

#ifdef BPNN_TUI
#include <ftxui/dom/elements.hpp>
namespace ftxui {
class ScreenInteractive;
}
#endif

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

   Dashboard(const Dashboard &) = delete;
   Dashboard &operator=(const Dashboard &) = delete;

   /// Whether a terminal UI should be shown (tty and BPNN_TUI env override).
   [[nodiscard]] static bool wantsTui() noexcept;

   /// Drives the whole session: starts the trainer on a background thread,
   /// runs the interactive FTXUI event loop on the calling thread and prints
   /// a plain-text summary once the session ends (also on 'q'/Esc).
   void run(NNtrainer &trainer);

   void onPass(std::size_t pass, bool show, Net &net,
               const nndef::values_layer_t &inputVals,
               const nndef::values_layer_t &resultVals,
               const nndef::values_layer_t &targetVals) override;
   void onFinished(Net &net, double elapsedMs = 0) override;

private:
   /// Immutable snapshot of the training state, produced by the trainer thread
   /// under mutex_ and consumed by the UI thread for rendering.
   struct Snapshot
   {
      std::size_t pass{0};
      double avgError{0.5};
      double bestError{0.5};
      double elapsedMs{0.0};
      double passesPerSec{0.0};
      bool finished{false};
      bool interrupted{false};
      nndef::values_layer_t inputs;
      nndef::values_layer_t results;
      nndef::values_layer_t targets;
      std::vector<std::pair<std::size_t, double>> errorHistory;
      std::size_t correctCount{0};
      std::size_t totalCount{0};
      std::size_t renderEvery{50};
      std::chrono::steady_clock::time_point rateStamp{};
      std::size_t ratePass{0};
      bool rateValid{false};
   };

   void printSummary() const;
#ifdef BPNN_TUI
   ftxui::Element buildDocument(const Snapshot &snapshot, bool paused, int plotW,
                                int plotH);
#endif

   Snapshot snapshot_{};
   mutable std::mutex mutex_{};

   Net &net_;
   const TrainingData &trnData_;
   std::string inputFileName_;
   std::size_t renderEvery_{50};

#ifdef BPNN_TUI
   ftxui::ScreenInteractive *screen_{nullptr};
   std::atomic<bool> loopReady_{false};
   std::atomic<bool> requestedInterrupt_{false};
   std::atomic<bool> activate_{false};
#endif
};

} // namespace tui

#endif // TUI_H