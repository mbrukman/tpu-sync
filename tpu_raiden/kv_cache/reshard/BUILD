# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The store-owned reshard control plane (C++ port of the Python
# rpc/raiden_controller.py pool-reshard path; migration plan §4.1).

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

RESHARD_COPTS = [
    "-fno-strict-aliasing",
    "-fexceptions",
]

cc_library(
    name = "declaration_types",
    srcs = ["declaration_types.cc"],
    hdrs = ["declaration_types.h"],
    copts = RESHARD_COPTS,
    features = ["-use_header_modules"],
    deps = [
        "//tpu_raiden/kv_cache:raiden_id",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "framed_rpc",
    srcs = ["framed_rpc.cc"],
    hdrs = ["framed_rpc.h"],
    copts = RESHARD_COPTS,
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "work_unit_directory",
    srcs = ["work_unit_directory.cc"],
    hdrs = ["work_unit_directory.h"],
    copts = RESHARD_COPTS,
    features = ["-use_header_modules"],
    deps = [
        ":declaration_types",
        "//tpu_raiden/kv_cache:raiden_id",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "request_block_registry",
    srcs = ["request_block_registry.cc"],
    hdrs = ["request_block_registry.h"],
    copts = RESHARD_COPTS,
    features = ["-use_header_modules"],
    deps = [
        ":declaration_types",
        ":work_unit_directory",
        "//tpu_raiden/kv_cache:pool_layout",
        "//tpu_raiden/kv_cache:raiden_id",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "pool_reshard_planner",
    srcs = ["pool_reshard_planner.cc"],
    hdrs = ["pool_reshard_planner.h"],
    copts = RESHARD_COPTS,
    features = ["-use_header_modules"],
    deps = [
        ":declaration_types",
        ":request_block_registry",
        "//tpu_raiden/kv_cache:pool_layout",
        "//tpu_raiden/kv_cache:raiden_id",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "reshard_coordinator",
    srcs = ["reshard_coordinator.cc"],
    hdrs = ["reshard_coordinator.h"],
    copts = RESHARD_COPTS,
    features = ["-use_header_modules"],
    deps = [
        ":declaration_types",
        ":framed_rpc",
        ":pool_reshard_planner",
        ":request_block_registry",
        ":work_unit_directory",
        "//tpu_raiden/core:transfer_program_reshard",
        "//tpu_raiden/core/controller:raiden_controller",
        "//tpu_raiden/kv_cache:raiden_id",
        "//tpu_raiden/proto:transfer_program_cc_proto",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "reshard_service",
    srcs = ["reshard_service.cc"],
    hdrs = ["reshard_service.h"],
    copts = RESHARD_COPTS,
    features = ["-use_header_modules"],
    deps = [
        ":declaration_types",
        ":framed_rpc",
        ":request_block_registry",
        ":reshard_coordinator",
        ":work_unit_directory",
        "//tpu_raiden/core:transfer_program_reshard",
        "//tpu_raiden/core/controller:raiden_controller",
        "//tpu_raiden/proto:transfer_program_cc_proto",
        "//tpu_raiden/rpc:controller_service_cc_proto",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_binary(
    name = "reshard_sidecar",
    srcs = ["reshard_sidecar_main.cc"],
    copts = RESHARD_COPTS,
    features = ["-use_header_modules"],
    deps = [
        ":reshard_service",
        "//tpu_raiden/kv_cache:kv_cache_store",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_binary(
    name = "replay_differential",
    srcs = ["replay_differential_main.cc"],
    copts = RESHARD_COPTS,
    features = ["-use_header_modules"],
    deps = [
        ":framed_rpc",
        ":reshard_service",
        "//tpu_raiden/core:transfer_program_reshard",
        "//tpu_raiden/proto:transfer_program_cc_proto",
        "//tpu_raiden/rpc:controller_service_cc_proto",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:differencer",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "reshard_service_test",
    srcs = ["reshard_service_test.cc"],
    copts = RESHARD_COPTS,
    features = ["-use_header_modules"],
    # Standalone binary: runnable straight from bazel-out without the
    # runfiles solib tree (the phase-A workflow executes it outside bazel).
    linkstatic = True,
    deps = [
        ":declaration_types",
        ":framed_rpc",
        ":pool_reshard_planner",
        ":request_block_registry",
        ":reshard_coordinator",
        ":reshard_service",
        ":work_unit_directory",
        "//tpu_raiden/rpc:controller_service_cc_proto",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "reshard_phaseb_test",
    srcs = ["reshard_phaseb_test.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    linkstatic = True,
    deps = [
        ":declaration_types",
        ":framed_rpc",
        ":reshard_coordinator",
        ":reshard_service",
        "//tpu_raiden/core:kv_manager_holder",
        "//tpu_raiden/core:raw_transfer_core",
        "//tpu_raiden/core:transfer_program_reshard",
        "//tpu_raiden/core/controller:raiden_controller",
        "//tpu_raiden/core/controller:worker_registry",
        "//tpu_raiden/core/controller:worker_service_client",
        "//tpu_raiden/core/controller:worker_service_server",
        "//tpu_raiden/kv_cache:raiden_id",
        "//tpu_raiden/proto:transfer_program_cc_proto",
        "//tpu_raiden/rpc:controller_service_cc_proto",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
