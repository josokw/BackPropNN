#include "Tui.h"

#include "AppInfo.h"
#include "NNconfig.h"
#include "Net.h"
#include "TrainingData.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#ifdef BPNN_TUI
#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/canvas.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/dom/node.hpp>
#include <ftxui/screen/color.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/screen/terminal.hpp>
#endif

namespace tui {

namespace {

#ifdef BPNN_TUI
constexpr std::size_t MAX_HISTORY_POINTS = 600;

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

/// Viridis-like colour ramp for t in [0.0, 1.0].
ftxui::Color viridis(double t)
{
   using ftxui::Color;
   t = std::clamp(t, 0.0, 1.0);
   if (t < 0.25) {
      return Color::Interpolate(static_cast<float>(t * 4.0),
                                Color::RGB(68, 1, 84), Color::RGB(59, 82, 139));
   }
   if (t < 0.5) {
      return Color::Interpolate(static_cast<float>((t - 0.25) * 4.0),
                                Color::RGB(59, 82, 139),
                                Color::RGB(33, 145, 140));
   }
   if (t < 0.75) {
      return Color::Interpolate(static_cast<float>((t - 0.5) * 4.0),
                                Color::RGB(33, 145, 140),
                                Color::RGB(94, 201, 98));
   }
   return Color::Interpolate(static_cast<float>((t - 0.75) * 4.0),
                             Color::RGB(94, 201, 98), Color::RGB(253, 231, 37));
}

/// Horizontal two-sided bar marking \p value and \p target.
ftxui::Element outBar(double value, double target)
{
   using namespace ftxui;
   constexpr int N = 34;
   double lo = std::min(value, target);
   double hi = std::max(value, target);
   if (lo > 0.0) {
      lo = 0.0;
   }
   if (hi < 0.0) {
      hi = 0.0;
   }
   const double span = (hi - lo) > 1e-12 ? (hi - lo) : 1.0;
   const auto col = [&](double v) {
      return static_cast<int>(std::lround((v - lo) / span *
                                          static_cast<double>(N - 1)));
   };

   Elements parts;
   parts.reserve(static_cast<std::size_t>(N));
   for (int i = 0; i < N; ++i) {
      auto e = text("·") | dim;
      if ((lo <= 0.0 && 0.0 <= hi) && i == col(0.0)) {
         e = text("│") | color(Color::GrayDark);
      }
      if (i == col(target)) {
         e = text("▎") | color(Color::Yellow) | bold;
      }
      if (i == col(value)) {
         e = text("█") | color(Color::Cyan) | bold;
      }
      if (i == col(value) and i == col(target)) {
         e = text("█") | color(Color::Magenta) | bold;
      }
      parts.push_back(e);
   }
   return hbox(std::move(parts)) | size(WIDTH, EQUAL, N);
}

/// Error-history line plot on a braille canvas (semi-log y axis).
ftxui::Element errorPlot(
   const std::vector<std::pair<std::size_t, double>> &history,
   double bestError, int W, int H)
{
   using namespace ftxui;
   const auto clampI = [](int v, int a, int b) { return std::clamp(v, a, b); };

   return canvas(W, H, [history, bestError, clampI, W, H](Canvas &c) {
      if (history.empty()) {
         return;
      }

      const Color trace{Color::RGB(33, 145, 140)};

      const auto finiteOf = [](double v) {
         return std::isfinite(v) ? v : 0.5;
      };
      const auto minErr = std::min_element(
         history.begin(), history.end(),
         [&](const auto &a, const auto &b) {
            return finiteOf(a.second) < finiteOf(b.second);
         });
      const auto maxErr = std::max_element(
         history.begin(), history.end(),
         [&](const auto &a, const auto &b) {
            return finiteOf(a.second) < finiteOf(b.second);
         });

      const double minY =
         std::log10(std::max(1e-9, minErr->second));
      const double maxY =
         std::log10(std::max(1e-9, maxErr->second));
      const double targetY = std::log10(MIN_RECENT_AVERAGE_ERROR);
      const double bestY = std::log10(std::max(1e-9, bestError));
      const double lo = std::min({minY, targetY, bestY});
      const double range = std::max(maxY - lo, 1e-9);

      const auto yOf = [&](double err) {
         return clampI(
            static_cast<int>((1.0 - (std::log10(std::max(1e-9, err)) - lo) /
                                       range) *
                             static_cast<double>(H - 1)),
            0, H - 1);
      };

      // Dashed line at the error target.
      const int yTarget = yOf(MIN_RECENT_AVERAGE_ERROR);
      for (int x = 0; x < W; x += 4) {
         c.DrawPointLine(x, yTarget, std::min(x + 2, W - 1), yTarget,
                         Color::Yellow);
      }
      // Dashed line at the best error reached so far.
      const int yBest = yOf(bestError);
      if (yBest != yTarget) {
         for (int x = 0; x < W; x += 4) {
            c.DrawPointLine(x, yBest, std::min(x + 2, W - 1), yBest,
                            Color::Magenta);
         }
      }

      const auto [firstPass, lastPass] =
         std::minmax_element(history.begin(), history.end(),
                             [](const auto &a, const auto &b) {
                                return a.first < b.first;
                             });
      const std::size_t span =
         std::max<std::size_t>(1, lastPass->first - firstPass->first);
      const auto xOf = [&](std::size_t pass) {
         return clampI(
            static_cast<int>(static_cast<double>(pass - firstPass->first) /
                             static_cast<double>(span) *
                             static_cast<double>(W - 1)),
            0, W - 1);
      };

      // Area fill under the curve, then a crisp trace on top.
      for (const auto &[pass, err] : history) {
         const int x = xOf(pass);
         const int y = yOf(err);
         for (int yy = y; yy < H; ++yy) {
            c.DrawPoint(x, yy, true, trace);
         }
      }
      int prevX = -1;
      int prevY = -1;
      for (const auto &[pass, err] : history) {
         const int x = xOf(pass);
         const int y = yOf(err);
         if (x == prevX and y == prevY) {
            continue;
         }
         if (prevX >= 0) {
            c.DrawPointLine(prevX, prevY, x, y, Color::White);
         }
         prevX = x;
         prevY = y;
      }

      // Highlight the latest history point.
      for (int dx = -1; dx <= 1; ++dx) {
         c.DrawPoint(prevX + dx, prevY, true, Color::Green);
      }
      for (int dy = -1; dy <= 1; ++dy) {
         c.DrawPoint(prevX, prevY + dy, true, Color::Green);
      }
   });
}

/// Two-character heatmap cell for a value in [0.0, 1.0].
ftxui::Element heatCell(double t)
{
   using namespace ftxui;
   return text(" ") | bgcolor(viridis(t)) | size(WIDTH, EQUAL, 2);
}

/// Grid of heatmap cells laid out as \p cols columns wide and \p rows tall.
ftxui::Element heatGrid(const std::vector<double> &v, std::size_t cols,
                        std::size_t rows, double minV, double maxV)
{
   using namespace ftxui;
   const double span = (maxV - minV) > 0.0 ? (maxV - minV) : 1.0;
   Elements rowsElems;
   rowsElems.reserve(rows);
   for (std::size_t r = 0; r < rows; ++r) {
      Elements cells;
      cells.reserve(cols);
      for (std::size_t c = 0; c < cols; ++c) {
         const std::size_t idx = r * cols + c;
         const double t = idx < v.size() ? (v[idx] - minV) / span : 0.0;
         cells.push_back(heatCell(t));
      }
      rowsElems.push_back(hbox(std::move(cells)));
   }
   return vbox(std::move(rowsElems));
}

/// Learned first-layer feature maps: a mini viridis heatmap per hidden neuron.
/// Weights are symmetric around 0 (0 = neutral teal), scaled by max |w|.
ftxui::Element featuresPanel(const Dashboard::Snapshot &s,
                             std::size_t gridCols)
{
   using namespace ftxui;
   constexpr std::size_t MAX_NEURONS = 12;

   if (s.features.empty()) {
      return window(text("Features L0-L1") | bold | color(Color::Cyan),
                    text("no first-layer weights yet") | dim);
   }

   Elements rows;
   const std::size_t nShown =
      std::min(MAX_NEURONS, s.features.size());
   rows.reserve(nShown + 2);
   for (std::size_t n = 0; n < nShown; ++n) {
      const auto &wv = s.features[n];
      std::size_t gcols = gridCols > 0 ? gridCols : wv.size();
      gcols = std::max<std::size_t>(1, gcols);
      const std::size_t grows = (wv.size() + gcols - 1) / gcols;
      rows.push_back(hbox({
         text(std::format("h{:>2}", n)) | dim | size(WIDTH, EQUAL, 4),
         heatGrid(wv, gcols, grows, -s.featuresScale, s.featuresScale),
      }));
   }
   rows.push_back(separator());
   rows.push_back(
      hbox({text("scale (0 = teal): ") | dim,
            text(std::format("{:.3f}", s.featuresScale)) | bold}));
   if (s.features.size() > nShown) {
      rows.push_back(text("... " +
                          std::to_string(s.features.size() - nShown) +
                          " more") |
                     dim);
   }
   return window(
      text(std::format("Features L0-L1  {} neurons", s.features.size())) |
         bold | color(Color::Cyan),
      vbox(std::move(rows)));
}
#endif

} // namespace

Dashboard::Dashboard(Net &net, const TrainingData &trnData,
                     std::string inputFileName, std::size_t renderEvery)
   : net_{net}
   , trnData_{trnData}
   , inputFileName_{std::move(inputFileName)}
   , renderEvery_{std::max<std::size_t>(renderEvery, 1)}
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
   (void)show;
#ifdef BPNN_TUI
   if (not activate_.load(std::memory_order_acquire)) {
      return;
   }
   Snapshot s;
   {
      std::lock_guard<std::mutex> lock(mutex_);
      s = snapshot_;
   }

   s.pass = pass;
   s.inputs = inputVals;
   s.results = resultVals;
   s.targets = targetVals;
   s.avgError = net.getRecentAverageError();
   s.bestError = std::min(s.bestError, s.avgError);

   // Fan-in weights of the first hidden layer, one vector per hidden neuron,
   // excluding the input layer's bias neuron. Pure reads of the trainer's own
   // thread; the UI thread only ever sees the snapshot.
   const auto &layers = net.layers();
   if (layers.size() >= 2) {
      const auto &inputLayer = layers[0];
      const auto &firstHidden = layers[1];
      std::vector<std::vector<double>> features;
      features.reserve(firstHidden.size());
      double maxAbs = 0.0;
      for (std::size_t n = 0; n + 1 < firstHidden.size(); ++n) {
         const auto &weights = firstHidden[n].getOutputWeights();
         std::vector<double> wv;
         wv.reserve(inputLayer.size() - 1);
         for (std::size_t p = 0; p + 1 < inputLayer.size(); ++p) {
            const double w = weights[p].weight;
            wv.push_back(w);
            maxAbs = std::max(maxAbs, std::abs(w));
         }
         features.push_back(std::move(wv));
      }
      s.features = std::move(features);
      s.featuresScale = maxAbs > 0.0 ? maxAbs : 1.0;
   }

   ++s.totalCount;
   if (resultVals.size() == targetVals.size() and not resultVals.empty() and
       argmaxIndex(resultVals) == argmaxIndex(targetVals)) {
      ++s.correctCount;
   }

   // Down-sampled rolling error history (decimate when it gets too large).
   if (pass % s.renderEvery == 0) {
      s.errorHistory.emplace_back(pass, s.avgError);
      while (s.errorHistory.size() > MAX_HISTORY_POINTS) {
         std::vector<std::pair<std::size_t, double>> decimated;
         decimated.reserve((s.errorHistory.size() + 1) / 2);
         for (std::size_t i = 0; i < s.errorHistory.size(); i += 2) {
            decimated.push_back(s.errorHistory[i]);
         }
         s.errorHistory.swap(decimated);
      }
   }

   // Training speed (passes per second), updated ~once per second.
   const auto now = std::chrono::steady_clock::now();
   const double dt = std::chrono::duration<double>(now - s.rateStamp).count();
   if (not s.rateValid) {
      s.rateValid = true;
      s.rateStamp = now;
      s.ratePass = pass;
   } else if (dt >= 1.0) {
      s.passesPerSec = static_cast<double>(pass - s.ratePass) / dt;
      s.rateStamp = now;
      s.ratePass = pass;
   }

   {
      std::lock_guard<std::mutex> lock(mutex_);
      snapshot_ = std::move(s);
   }
   revision_.fetch_add(1, std::memory_order_relaxed);
#endif
}

void Dashboard::onFinished(Net &net, double elapsedMs)
{
   (void)net;
   {
      std::lock_guard<std::mutex> lock(mutex_);
      snapshot_.finished = true;
      snapshot_.elapsedMs = elapsedMs;
#ifdef BPNN_TUI
       snapshot_.interrupted = requestedInterrupt_.load();
#endif
   }
}

void Dashboard::printSummary() const
{
   Snapshot s;
   {
      std::lock_guard<std::mutex> lock(mutex_);
      s = snapshot_;
   }
   std::cout << std::fixed;
   std::cout << "\n- Training " << (s.interrupted ? "(interrupted) " : "done")
             << ": " << s.pass << " passes, avg error " << std::setprecision(6)
             << s.avgError << ", best error " << s.bestError;
   if (s.elapsedMs > 0.0) {
      std::cout << std::setprecision(1) << ", " << s.elapsedMs << " ms";
   }
   if (s.totalCount > 0) {
      std::cout << std::setprecision(1)
                << ", accuracy "
                << 100.0 * static_cast<double>(s.correctCount) /
                      static_cast<double>(s.totalCount)
                << "% (" << s.correctCount << "/" << s.totalCount << ")";
   }
   std::cout << "\n";
}

#ifdef BPNN_TUI
ftxui::Element Dashboard::buildDocument(const Snapshot &s, bool paused, int W,
                                        int H, bool showHelp)
{
   using namespace ftxui;

   const auto &topology = trnData_.getTopology();
   const auto &actNames = trnData_.getActionFunctionNames();

   const auto panel = [](const std::string &title) {
      return text(title) | bold | color(Color::Cyan);
   };
   const auto row = [](const std::string &name, const std::string &value) {
      return hbox({text(name) | dim, filler(), text(value) | bold});
   };

   // --- status badge ---------------------------------------------------
   const std::string badge =
      s.finished ? "DONE" : (paused ? "PAUSED" : "RUNNING");
   const auto badgeColor =
      s.finished ? Color::Green : (paused ? Color::Yellow : Color::Cyan);

   // --- header ----------------------------------------------------------
   const auto header = vbox({
      hbox({
         text(" " APPNAME_VERSION " ") | bold | color(Color::Green),
         separator(),
         text("live training: ") | dim,
         text(inputFileName_) | bold | color(Color::Yellow),
         filler(),
         text(badge) | bold | color(badgeColor),
      }),
      hbox({
         text(std::format("pass {:>7}   speed {:>7.0f}/s   err {:.6f}", s.pass,
                          s.passesPerSec, s.avgError)) |
            dim,
         filler(),
         text((paused ? "paused" : "running")) | dim,
      }),
   });

   // --- settings ---------------------------------------------------------
   std::vector<Element> layers;
   for (std::size_t i = 0; i < topology.size(); ++i) {
      const auto kind =
         (i + 1 == topology.size()) ? "output" : (i == 0 ? "input" : "hidden");
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
         separator(),
         text("every: " + std::to_string(s.renderEvery) + " passes") | dim,
      }));

   // --- training / progress --------------------------------------------
   const double passFrac = std::clamp(
      static_cast<double>(s.pass) / static_cast<double>(MAX_ITERATIONS), 0.0,
      1.0);
   const double errFrac = std::clamp(
      (0.5 - s.avgError) / (0.5 - MIN_RECENT_AVERAGE_ERROR), 0.0, 1.0);

   std::string eta;
   if (not s.finished) {
      const double remaining =
         s.avgError > MIN_RECENT_AVERAGE_ERROR
            ? static_cast<double>(MAX_ITERATIONS - s.pass)
            : 0.0;
      if (s.passesPerSec > 0.0 and remaining > 0.0) {
         const auto seconds = static_cast<std::size_t>(
            std::max(0.0, remaining / s.passesPerSec));
         eta = std::format("{:02}:{:02}:{:02}", seconds / 3600,
                           (seconds / 60) % 60, seconds % 60);
      } else {
         eta = "-";
      }
   }

   std::vector<Element> trainingRows{
      row("Pass", std::format("{} / {}", s.pass, MAX_ITERATIONS)),
      hbox({text("Iterations") | dim, filler(),
            gauge(passFrac) | color(Color::Blue)}),
      row("Average error", std::format("{:.6f}", s.avgError)),
      row("Best error", std::format("{:.6f}", s.bestError)),
      hbox({text("Error target") | dim, filler(),
            gauge(errFrac) | color(Color::Green)}),
   };
   if (not s.finished) {
      trainingRows.push_back(row("ETA to max", eta));
   }
   if (s.finished) {
      trainingRows.push_back(
         row("Elapsed", std::format("{:.1f} ms", s.elapsedMs)));
   }
   const auto training =
      window(panel(s.finished ? "Training (done)" : "Training"),
             vbox(std::move(trainingRows)));

   // --- responsive sizes (recomputed every frame) ------------------------
   const int plotH = std::clamp(H - 40, 5, 24);
   const int plotW = std::clamp(W - 6, 40, 240);
   const int settingsW = W >= 90 ? 34 : std::max(24, W / 3);
   const std::size_t gridCols =
      trnData_.show_max_inputs > 0
         ? static_cast<std::size_t>(trnData_.show_max_inputs)
         : static_cast<std::size_t>(
              std::sqrt(std::max<double>(1.0, s.inputs.size())));

   // --- error history plot ---------------------------------------------
   double histMin = s.bestError;
   double histMax = std::max(0.5, s.bestError);
   if (not s.errorHistory.empty()) {
      const auto [mn, mx] = std::minmax_element(
         s.errorHistory.begin(), s.errorHistory.end(),
         [](const auto &a, const auto &b) { return a.second < b.second; });
      histMin = mn->second;
      histMax = mx->second;
   }
   const auto plot = window(
      panel(std::format("Error history (log)  avg {:.6f}  best {:.6f}",
                        s.avgError, s.bestError)),
      vbox({
         errorPlot(s.errorHistory, s.bestError, plotW, plotH) | border,
         hbox({
            text(std::format("range [{:.4f}, {:.4f}]", histMin, histMax)) |
               dim,
            filler(),
            text(std::format("target {:.3f}", MIN_RECENT_AVERAGE_ERROR)) |
               color(Color::Yellow),
            text(" best ") | color(Color::Magenta),
            text(" now ") | color(Color::Green),
         }),
      }));

   // --- current sample ---------------------------------------------------
   std::vector<Element> sampleRows;
   if (not s.inputs.empty()) {
      const auto [loIt, hiIt] =
         std::minmax_element(s.inputs.begin(), s.inputs.end());
      const double gridMin = *loIt;
      const double gridMax = *hiIt;
      const std::size_t grows =
         (s.inputs.size() + gridCols - 1) / gridCols;
      sampleRows.push_back(
         hbox({heatGrid(s.inputs, gridCols, grows, gridMin, gridMax)}) |
         size(WIDTH, LESS_THAN, gridCols * 2 + 4));
   }

   // --- output bars --------------------------------------------------------
   std::vector<Element> outRows;
   const auto nTargets = std::min(s.targets.size(), s.results.size());
   for (std::size_t o = 0; o < nTargets; ++o) {
      std::string name = (o < trnData_.output_names.size())
                            ? trnData_.output_names[o]
                            : std::format("out {}", o);
      outRows.push_back(hbox({
         text(std::format("{:>8}", name)) | dim | size(WIDTH, EQUAL, 9),
         outBar(s.results[o], s.targets[o]) | flex,
         text(std::format("{:+.3f}", s.results[o])) |
            size(WIDTH, EQUAL, 8) | color(Color::Cyan) | bold,
      }));
   }

   const bool correct = not s.results.empty() and
                        s.results.size() == s.targets.size() and
                        argmaxIndex(s.results) == argmaxIndex(s.targets);
   std::string className;
   if (not s.targets.empty()) {
      const auto idx = argmaxIndex(s.targets);
      if (idx < trnData_.output_names.size()) {
         className = trnData_.output_names[idx];
      }
   }

   sampleRows.push_back(separator());
   sampleRows.push_back(hbox({
      text("targets: ") | dim,
      text(join(s.targets)) | color(Color::Yellow),
   }));
   sampleRows.push_back(
      hbox({text("outputs: ") | dim, text(join(s.results))}));
   if (not outRows.empty()) {
      sampleRows.push_back(separator());
      sampleRows.push_back(vbox(std::move(outRows)));
   }
   sampleRows.push_back(hbox({
      text("accuracy: ") | dim,
      text(std::format("{} / {} ({:.1f}%)", s.correctCount, s.totalCount,
                       100.0 * static_cast<double>(s.correctCount) /
                          std::max<std::size_t>(1, s.totalCount))) |
         color(Color::Green),
   }));
   sampleRows.push_back(separator());
   if (s.finished) {
sampleRows.push_back(text("done — press q to exit (summary follows)") |
                            dim);
   } else {
      sampleRows.push_back(
         correct
            ? text(std::format("class '{}'  correct", className)) |
                 color(Color::Green)
            : text(std::format("class '{}'  miss", className)) |
                 color(Color::Red));
   }

   const auto sample = window(panel("Current sample"),
                              vbox(std::move(sampleRows)));

   // --- learned features ------------------------------------------------
   Element features = text("");
   if (W >= 130 and not s.features.empty()) {
      const std::size_t fanIn =
         topology.size() >= 2 ? topology[0] : s.inputs.size();
      std::size_t fcols =
         trnData_.show_max_inputs > 0
            ? static_cast<std::size_t>(trnData_.show_max_inputs)
            : static_cast<std::size_t>(
                 std::sqrt(std::max<double>(1.0, fanIn)));
      features = featuresPanel(s, fcols) | border;
   }

   const auto footer = hbox({
      text(" q quit ") | color(Color::White) | bgcolor(Color::GrayDark),
      text(" p pause ") | color(Color::White) | bgcolor(Color::GrayDark),
      text(" h help ") | color(Color::White) | bgcolor(Color::GrayDark),
      text(" +/- render ") | color(Color::White) | bgcolor(Color::GrayDark),
      text(" scroll ") | color(Color::White) | bgcolor(Color::GrayDark),
      filler(),
      text((paused ? "paused" : "running")) | dim,
   });

   // --- assemble the scrollable dashboard body --------------------------
   const auto body = vbox({
      header | border,
      hbox({
         settings | size(WIDTH, EQUAL, settingsW) | border,
         separator(),
         vbox({training, separator(), sample}) | flex,
         features,
      }),
      separator(),
      plot | size(HEIGHT, EQUAL, plotH + 4),
   });

   const int viewH = std::max(2, H - 3); // footer (~3 rows incl. border) pinned.
   body->ComputeRequirement();
   const int contentMinY =
      std::max(viewH, static_cast<int>(body->requirement().min_y));
   const int maxOffset = std::max(0, contentMinY - viewH);
   const int offset = std::min(scrollOffset_, maxOffset);
   // focusPosition puts the chosen row at the viewport centre; FTXUI frames
   // math uses (viewH-1)/2 and size(viewH-1), so request offset + (viewH-1)/2
   // to make exactly `offset` content rows pan out of view.
   const int yAnchor = offset + (viewH - 1) / 2;
   const auto scrolled =
      body | focusPosition(0, yAnchor) | vscroll_indicator | frame |
      size(HEIGHT, EQUAL, viewH);

   Element doc = vbox({scrolled, footer | border});

   // --- help overlay ----------------------------------------------------
   if (showHelp) {
      const auto helpBox = window(
         panel("Keys"),
         vbox({
            row("q / Esc", "stop training and exit"),
            row("p / Space", "pause / resume"),
            row("h / ?", "toggle this help"),
            row("+ / -", "halve / double render cadence"),
            row("up / down", "scroll 1 line"),
            row("PgUp / PgDn", "scroll half a page"),
            row("Home / End", "jump to top / bottom"),
            row("mouse wheel", "scroll"),
         })) |
         size(WIDTH, EQUAL, 48) | center;
      doc = dbox({doc, helpBox});
   }

   return doc;
}

void Dashboard::run(NNtrainer &trainer)
{
   using namespace ftxui;

   auto screen = ScreenInteractive::Fullscreen();
   screen_ = &screen;

   {
      std::lock_guard<std::mutex> lock(mutex_);
      snapshot_ = Snapshot{};
      snapshot_.renderEvery = renderEvery_;
   }
   activate_.store(true, std::memory_order_relaxed);
   revision_.store(0, std::memory_order_relaxed);

   std::atomic<bool> keepRefreshing{true};
   std::thread trainerThread([this, &trainer] {
      trainer.setPaused(false);
      trainer.train();
      activate_.store(false, std::memory_order_relaxed);
   });
   // Paints only when the trainer published a new snapshot; while paused the
   // revision never changes, so an idle (paused / nothing new) dashboard does
   // no busy repainting.
   std::size_t lastRevision = 0;
   std::thread refresher([this, &keepRefreshing, &lastRevision] {
      while (keepRefreshing.load(std::memory_order_acquire)) {
         std::this_thread::sleep_for(std::chrono::milliseconds(100));
         if (not keepRefreshing.load(std::memory_order_acquire)) {
            break;
         }
         if (screen_ and loopReady_.load(std::memory_order_acquire) and
             lastRevision != revision_.load(std::memory_order_acquire)) {
            lastRevision =
               revision_.load(std::memory_order_acquire);
            screen_->PostEvent(Event::Custom);
         }
      }
   });

   const auto scrollBy = [this](int delta) {
      const int cur = static_cast<int>(scrollOffset_);
      scrollOffset_ = static_cast<std::size_t>(std::max(0, cur + delta));
   };

   auto component = CatchEvent(
      Renderer([this, &trainer] {
         const auto dim = ftxui::Terminal::Size();
         Snapshot s;
         {
            std::lock_guard<std::mutex> lock(mutex_);
            s = snapshot_;
         }
         return buildDocument(s, trainer.isPaused(),
                              static_cast<int>(dim.dimx),
                              static_cast<int>(dim.dimy), showHelp_);
      }),
      [this, &trainer, &keepRefreshing, &trainerThread, &refresher,
       &scrollBy](Event event) {
         // Only the main loop thread ever exits the screen, and the worker
         // threads are joined first so no PostEvent from a background thread
         // can race with FTXUI's internal teardown.
         if (event == Event::Custom) {
            bool finished = false;
            {
               std::lock_guard<std::mutex> lock(mutex_);
               finished = snapshot_.finished;
            }
            if (finished) {
               // Training is done: stop repainting, but keep the final
               // DONE dashboard on screen until the user quits with q/Escape.
               keepRefreshing.store(false, std::memory_order_relaxed);
               if (refresher.joinable()) {
                  refresher.join();
               }
            }
            return true;
         }
         // --- scrolling ---------------------------------------------------
         if (event == Event::ArrowDown or event == Event::Character('j')) {
            scrollBy(1);
            return true;
         }
         if (event == Event::ArrowUp or event == Event::Character('k')) {
            scrollBy(-1);
            return true;
         }
         if (event == Event::PageDown) {
            scrollBy(12);
            return true;
         }
         if (event == Event::PageUp) {
            scrollBy(-12);
            return true;
         }
         if (event == Event::Home) {
            scrollOffset_ = 0;
            return true;
         }
         if (event == Event::End) {
            scrollOffset_ = std::numeric_limits<int>::max();
            return true;
         }
         if (event.is_mouse()) {
            auto &mouse = event.mouse();
            if (mouse.button == Mouse::WheelDown) {
               scrollBy(3);
               return true;
            }
            if (mouse.button == Mouse::WheelUp) {
               scrollBy(-3);
               return true;
            }
         }
         if (event == Event::Character('h') or event == Event::Character('?')) {
            showHelp_ = not showHelp_;
            return true;
         }
         if (event == Event::Character('q') or event == Event::Escape) {
            requestedInterrupt_.store(true, std::memory_order_relaxed);
            trainer.requestStop();
            keepRefreshing.store(false, std::memory_order_relaxed);
            if (trainerThread.joinable()) {
               trainerThread.join();
            }
            if (refresher.joinable()) {
               refresher.join();
            }
            if (screen_) {
               screen_->Exit();
            }
            return true;
         }
         if (event == Event::Character('p') or
             event == Event::Character(' ')) {
            trainer.setPaused(not trainer.isPaused());
            return true;
         }
         if (event == Event::Character('+') or
             event == Event::Character('=')) {
            renderEvery_ = std::min<std::size_t>(renderEvery_ * 2, 100'000);
            std::lock_guard<std::mutex> lock(mutex_);
            snapshot_.renderEvery = renderEvery_;
            return true;
         }
         if (event == Event::Character('-') or event == Event::Character('_')) {
            renderEvery_ = std::max<std::size_t>(1, renderEvery_ / 2);
            std::lock_guard<std::mutex> lock(mutex_);
            snapshot_.renderEvery = renderEvery_;
            return true;
         }
         return false;
      });

   // Destroying a joinable std::thread aborts the process ("terminate called
   // without an active exception"). If the FTXUI loop throws (terminal error,
   // rendering failure) the worker threads are unwound out of scope, so join
   // them first and never destroy a live thread.
   try {
      loopReady_.store(true, std::memory_order_release);
      screen.Loop(std::move(component));
   } catch (...) {
      loopReady_.store(false, std::memory_order_relaxed);
      keepRefreshing.store(false, std::memory_order_relaxed);
      if (trainerThread.joinable()) {
         trainerThread.join();
      }
      if (refresher.joinable()) {
         refresher.join();
      }
      screen_ = nullptr;
      activate_.store(false, std::memory_order_relaxed);
      throw;
   }
   loopReady_.store(false, std::memory_order_relaxed);

   keepRefreshing.store(false, std::memory_order_relaxed);
   if (trainerThread.joinable()) {
      trainerThread.join();
   }
   if (refresher.joinable()) {
      refresher.join();
   }

   printSummary();
   screen_ = nullptr;
   activate_.store(false, std::memory_order_relaxed);
}
#endif

} // namespace tui