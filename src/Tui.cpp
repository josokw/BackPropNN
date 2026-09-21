#include "Tui.h"

#include "AppInfo.h"
#include "NNconfig.h"
#include "Net.h"
#include "TrainingData.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#ifdef BPNN_TUI
#include <ftxui/dom/elements.hpp>
#include <ftxui/dom/node.hpp>
#include <ftxui/screen/screen.hpp>
#endif

namespace tui {

namespace {

std::string join(const nndef::values_layer_t &v, int precision = 3)
{
   std::ostringstream os;
   os << std::showpos << std::fixed << std::setprecision(precision);
   for (const auto &x : v) {
      os << std::setw(6) << x << " ";
   }
   return os.str();
}

std::size_t argmaxIndex(const nndef::values_layer_t &v)
{
   if (v.empty()) {
      return 0;
   }
   return static_cast<std::size_t>(
      std::distance(v.begin(), std::max_element(v.begin(), v.end())));
}

} // namespace

Dashboard::Dashboard(Net &net, const TrainingData &trnData,
                     std::string inputFileName, std::size_t renderEvery)
   : net_{net}
   , trnData_{trnData}
   , inputFileName_{std::move(inputFileName)}
   , renderEvery_{std::max<std::size_t>(renderEvery, 1)}
   , lastInputs_{}
   , lastResults_{}
   , lastTargets_{}
   , resetPosition_{}
{
}

bool Dashboard::wantsTui() noexcept
{
#ifdef BPNN_TUI
   if (const char *env = std::getenv("BPNN_TUI")) {
      const std::string value{env};
      return value != "0" and value != "off" and value != "false" and
             value != "no";
   }
   return ::isatty(::fileno(stdout)) == 1;
#else
   return false;
#endif
}

void Dashboard::onPass(std::size_t pass, bool show, Net &net,
                       const nndef::values_layer_t &inputVals,
                       const nndef::values_layer_t &resultVals,
                       const nndef::values_layer_t &targetVals)
{
   lastPass_ = pass;
   lastInputs_ = inputVals;
   lastResults_ = resultVals;
   lastTargets_ = targetVals;
   if (show or lastPass_ % renderEvery_ == 0) {
      render(net);
   }
}

void Dashboard::onFinished(Net &net, double elapsedMs)
{
   finished_ = true;
   elapsedMs_ = elapsedMs;
   render(net);
}

void Dashboard::render(Net &net)
{
#ifdef BPNN_TUI
   using namespace ftxui;

   const auto &topology = trnData_.getTopology();
   const auto &actNames = trnData_.getActionFunctionNames();
   const auto &inputs = lastInputs_;
   const auto &results = lastResults_;
   const auto &targets = lastTargets_;

   const auto panel = [](const std::string &title) {
      return text(title) | bold | color(Color::Cyan);
   };
   const auto row = [](const std::string &name, const std::string &value) {
      return hbox({text(name) | dim, filler(), text(value) | bold});
   };

   // --- header bar -------------------------------------------------------
   const auto header = hbox({
      text(" " APPNAME_VERSION " ") | bold | color(Color::Green),
      separator(),
      text("live training: ") | dim,
      text(inputFileName_) | bold | color(Color::Yellow),
      filler(),
      text(std::format("pass {}   avg error {:.6f}", lastPass_,
                       net.getRecentAverageError())) | dim,
   });

   // --- settings panel -----------------------------------------------------
   std::vector<Element> layers;
   for (std::size_t i = 0; i < topology.size(); ++i) {
      const auto kind = (i + 1 == topology.size())
                           ? "output"
                           : (i == 0 ? "input" : "hidden");
      layers.push_back(row(std::format("L{} {}: {} neurons", i, kind,
                                       topology[i]),
                           actNames[i]));
   }
   const auto settings = window(
      panel("Settings"),
      vbox({
         row("ETA", std::format("{:.4f}", trnData_.ETA)),
         row("ALPHA", std::format("{:.4f}", trnData_.ALPHA)),
         row("Seed", std::to_string(trnData_.seed)),
         separator(),
         vbox(std::move(layers)),
      }));

   // --- training panel ------------------------------------------------------
   const double avgErr = net.getRecentAverageError();
   const double passFrac = std::clamp(
      static_cast<double>(lastPass_) / static_cast<double>(MAX_ITERATIONS),
      0.0, 1.0);
   const double errFrac = std::clamp(
      (0.5 - avgErr) / (0.5 - MIN_RECENT_AVERAGE_ERROR), 0.0, 1.0);
   std::vector<Element> trainingRows{
      row("Pass", std::format("{} / {}", lastPass_, MAX_ITERATIONS)),
      hbox({text("Iterations") | dim, filler(),
            gauge(passFrac) | color(Color::Blue)}),
      row("Average error", std::format("{:.6f}", avgErr)),
      hbox({text("Error target") | dim, filler(),
            gauge(errFrac) | color(Color::Green)}),
   };
   if (finished_) {
      trainingRows.push_back(
         row("Elapsed", std::format("{:.1f} ms", elapsedMs_)));
   }
   const auto training = window(
      panel(finished_ ? "Training (done)" : "Training"),
      vbox(std::move(trainingRows)));

   // --- current sample panel --------------------------------------------------
   const std::size_t gridCols =
      trnData_.show_max_inputs > 0
         ? static_cast<std::size_t>(trnData_.show_max_inputs)
         : static_cast<std::size_t>(
              std::sqrt(std::max<double>(1.0, inputs.size())));
   std::vector<Element> grid;
   if (not inputs.empty()) {
      const auto [low, high] =
         std::minmax_element(inputs.begin(), inputs.end());
      const double span = (*high - *low) > 0.0 ? (*high - *low) : 1.0;
      for (std::size_t i = 0; i < inputs.size();) {
         std::vector<Element> cells;
         for (std::size_t c = 0; c < gridCols and i < inputs.size();
              ++c, ++i) {
            const double t = (inputs[i] - *low) / span;
            const auto shade =
               static_cast<uint8_t>(15 + static_cast<int>(t * 235.0));
            cells.push_back(
               text("  ") | bgcolor(Color::RGB(shade, shade, shade)));
         }
         grid.push_back(hbox(std::move(cells)));
      }
   }

   const bool correct = not results.empty() and
                        results.size() == targets.size() and
                        argmaxIndex(results) == argmaxIndex(targets);
   std::string className;
   if (not targets.empty()) {
      const auto idx = argmaxIndex(targets);
      if (idx < trnData_.output_names.size()) {
         className = trnData_.output_names[idx];
      }
   }
   const auto sample = window(
      panel("Current sample"),
      vbox({
         vbox(std::move(grid)) | border,
         separator(),
         text("targets: " + join(targets)) | dim,
         text("outputs: " + join(results)),
         separator(),
         correct
            ? text(std::format("class '{}'  correct", className)) |
                 color(Color::Green)
            : text(std::format("class '{}'  miss", className)) |
                 color(Color::Red),
      }));

   const auto document = vbox({
      header | border,
      hbox({
         settings | size(WIDTH, EQUAL, 34) | border,
         separator(),
         vbox({training, separator(), sample}) | flex,
      }),
   });

   // Non-interactive in-place repaint (like the FTXUI 'gauge' example):
   auto screen = Screen::Create(Dimension::Full(), Dimension::Full());
   Render(screen, document | flex);
   std::cout << resetPosition_;
   screen.Print();
   resetPosition_ = screen.ResetPosition(/*clear=*/false);
#else
   (void)net;
#endif
}

} // namespace tui