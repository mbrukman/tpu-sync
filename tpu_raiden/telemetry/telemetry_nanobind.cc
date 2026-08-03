// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tpu_raiden/telemetry/telemetry_nanobind.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>  // IWYU pragma: keep
#include <nanobind/stl/vector.h>  // IWYU pragma: keep
#include "tpu_raiden/telemetry/metrics_api.h"
#ifndef TPU_RAIDEN_DISABLE_TELEMETRY
#include "tpu_raiden/telemetry/prometheus_exporter.h"
#endif

namespace nb = nanobind;

namespace tpu_raiden::telemetry {

void BindTelemetryApi(nb::module_& m) {
  m.def(
      "configure_telemetry",
      [](bool enabled, const std::vector<std::string>& backends) {
        auto& store = RaidenMetricStore::GetGlobalMetricStore();
        store.ClearBackends();
        if (!enabled) {
          return;
        }
        for (const auto& backend_name : backends) {
#ifndef TPU_RAIDEN_DISABLE_TELEMETRY
          if (backend_name == "prometheus") {
            store.AddBackend(std::make_unique<PrometheusExporter>());
          }
#endif
        }
      },
      nb::arg("enabled"), nb::arg("backends"),
      "Configures active C++ telemetry backends.");

  m.def(
      "is_telemetry_enabled",
      []() -> bool {
        return RaidenMetricStore::GetGlobalMetricStore().HasBackends();
      },
      "Returns true if C++ telemetry backends are actively enabled.");

  m.def(
      "get_raiden_metrics_prometheus_text",
      []() -> std::string {
        return RaidenMetricStore::GetGlobalMetricStore().GetTextSnapshot();
      },
      nb::call_guard<nb::gil_scoped_release>(),
      "Exports Prometheus text snapshot of TPU Raiden metrics.");
}

}  // namespace tpu_raiden::telemetry
