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

// Gate GB2 (phase B): the zero-sidecar control-plane topology, in-process.
//
//   source ReshardService (kController)   destination ReshardService
//     |  local metadata + planner            (kController, own port)
//     |  RaidenController A ── gRPC ──> fake sender workers (WorkerService)
//     |  framed arm relay ────────────> destination service
//     |                                   RaidenController B ── gRPC ──>
//     |                                     fake receiver worker
//
// Everything real except the KV managers: the fakes record the
// StartTransferRequest the holder lowers for them, so the test proves the
// door, carrier, relay, and coordinator wiring end-to-end, plus the failure
// matrix (§6 GB2): relay arm refusal -> claim abandoned + senders never
// contacted; unregistered destination worker -> retryable error at the
// source; framed-mode destination -> relay fail-closed.

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "tpu_raiden/core/controller/raiden_controller.h"
#include "tpu_raiden/core/controller/worker_registry.h"
#include "tpu_raiden/core/controller/worker_service_client.h"
#include "tpu_raiden/core/controller/worker_service_server.h"
#include "tpu_raiden/core/kv_manager_holder.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/core/transfer_program_reshard.h"
#include "tpu_raiden/kv_cache/raiden_id.h"
#include "tpu_raiden/kv_cache/reshard/declaration_types.h"
#include "tpu_raiden/kv_cache/reshard/reshard_coordinator.h"
#include "tpu_raiden/kv_cache/reshard/reshard_service.h"
#include "tpu_raiden/proto/transfer_program.pb.h"
#include "tpu_raiden/rpc/controller_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {
namespace {

namespace rpcpb = tpu_raiden::rpc;
using ::testing::HasSubstr;

RaidenId Unit(int rank) {
  RaidenId id;
  id.job_name = "prefill";
  id.job_replica_id = absl::StrCat(rank);
  id.data_name = "kv_cache";
  id.data_replica_idx = 0;
  return id;
}

RaidenId DstUnit() {
  RaidenId id;
  id.job_name = "decode";
  id.job_replica_id = "0";
  id.data_name = "kv_cache";
  id.data_replica_idx = 0;
  return id;
}

rpcpb::PoolSpecProto MakePool(const std::string& tag, int64_t live,
                              int64_t stride, int64_t num_blocks) {
  rpcpb::PoolSpecProto pool;
  pool.set_tag(tag);
  pool.set_storage_index(0);
  pool.set_base_offset_bytes(0);
  pool.set_block_stride_bytes(stride);
  pool.set_num_blocks(num_blocks);
  pool.set_dtype_tag("uint8");
  auto* region = pool.add_regions();
  region->set_name("live");
  region->set_offset_bytes(0);
  region->set_stride_bytes(live);
  region->set_unit_bytes(live);
  region->set_num_units(1);
  region->set_units_per_stride(1);
  return pool;
}

// Records the pool-executor calls the phase-B door lowers onto it. The
// non-pool KVManagerHolder ops are unreachable from this test and fail
// closed.
struct FakePoolManager {
  absl::Mutex mu;
  std::vector<rpcpb::StartTransferRequest> push_requests;
  std::vector<std::vector<int64_t>> push_ids;
  std::vector<int> push_parallelism;
  std::vector<rpcpb::StartTransferRequest> arm_requests;
  std::vector<std::vector<int64_t>> arm_ids;
  absl::Status push_result = absl::OkStatus();
  absl::Status arm_result = absl::OkStatus();

  absl::StatusOr<raiden::PjRtCopyFuture> D2h(const std::vector<int64_t>&,
                                             const std::vector<int64_t>&,
                                             const std::vector<int64_t>&) {
    return absl::UnimplementedError("fake manager");
  }
  absl::StatusOr<raiden::PjRtCopyFuture> H2d(const std::vector<int64_t>&,
                                             const std::vector<int64_t>&,
                                             const std::vector<int64_t>&) {
    return absl::UnimplementedError("fake manager");
  }
  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hRead(
      std::string, const std::vector<int>&) {
    return absl::UnimplementedError("fake manager");
  }
  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hWrite(
      std::string, const std::vector<int>&, const std::vector<int>&) {
    return absl::UnimplementedError("fake manager");
  }
  absl::Status PoolReshardPush(const rpcpb::StartTransferRequest& request,
                               absl::Span<const int64_t> src_block_ids,
                               int parallelism) {
    absl::MutexLock lock(&mu);
    if (!push_result.ok()) return push_result;
    push_requests.push_back(request);
    push_ids.emplace_back(src_block_ids.begin(), src_block_ids.end());
    push_parallelism.push_back(parallelism);
    return absl::OkStatus();
  }
  absl::Status PoolReshardRegisterRecv(
      const rpcpb::StartTransferRequest& request,
      absl::Span<const int64_t> chip_block_ids) {
    absl::MutexLock lock(&mu);
    if (!arm_result.ok()) return arm_result;
    arm_requests.push_back(request);
    arm_ids.emplace_back(chip_block_ids.begin(), chip_block_ids.end());
    return absl::OkStatus();
  }
};

// One fake worker: a private WorkerServiceServer wrapping a FakePoolManager.
struct FakeWorker {
  FakePoolManager manager;
  std::unique_ptr<controller::WorkerServiceServer> server;
  int port = 0;

  absl::Status Start() {
    server = controller::WorkerServiceServer::Create();
    absl::Status status = server->StartServer(
        /*host_allocator=*/nullptr, KVManagerHolder(&manager), /*port=*/0);
    if (!status.ok()) return status;
    port = server->GetRaidenWorkerPort();
    return absl::OkStatus();
  }
};

class ReshardPhaseBTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Two dispatch controllers coexist in this process; keep their
    // ControllerServers private.
    ::setenv("RAIDEN_DISABLE_SINGLETON_WORKER", "1", 1);
  }

  void StartWorkers(int num_src) {
    src_workers_.resize(num_src);
    for (int i = 0; i < num_src; ++i) {
      src_workers_[i] = std::make_unique<FakeWorker>();
      ASSERT_TRUE(src_workers_[i]->Start().ok());
    }
    ASSERT_TRUE(dst_worker_.Start().ok());
  }

  static std::unique_ptr<tpu_raiden::controller::RaidenController>
  MakeController(const RaidenId& unit) {
    rpcpb::RaidenIdProto unit_proto;
    unit_proto.set_job_name(unit.job_name);
    unit_proto.set_job_replica_id(unit.job_replica_id);
    unit_proto.set_data_name(unit.data_name);
    unit_proto.set_data_replica_idx(unit.data_replica_idx);
    return std::make_unique<tpu_raiden::controller::RaidenController>(
        unit_proto, /*num_blocks=*/0, /*num_shards=*/0,
        /*shard_size_bytes=*/0,
        /*raiden_orchestrator_address=*/"",
        /*raiden_controller_address=*/"127.0.0.1:0",
        /*preprovision_worker_buffers=*/false,
        /*expected_worker_count=*/0);
  }

  static void RegisterFakeWorker(
      tpu_raiden::controller::RaidenController* controller,
      const std::string& worker_id, int port, int64_t node_id) {
    core::controller::WorkerRegistration registration;
    registration.worker_id = worker_id;
    registration.raiden_worker_endpoint = absl::StrCat("127.0.0.1:", port);
    registration.worker_service_client =
        std::make_shared<tpu_raiden::controller::WorkerServiceClient>(
            grpc::CreateChannel(absl::StrCat("127.0.0.1:", port),
                                grpc::InsecureChannelCredentials()));
    registration.node_id = node_id;
    ASSERT_TRUE(
        controller->worker_registry()->RegisterWorker(registration).ok());
  }

  // Builds the two-service topology: source (drives planning + dispatch)
  // and destination (serves metadata + the arm relay on a real socket).
  void BuildTopology(int num_src) {
    StartWorkers(num_src);
    src_controller_ = MakeController(Unit(0));
    dst_controller_ = MakeController(DstUnit());
    for (int i = 0; i < num_src; ++i) {
      RegisterFakeWorker(src_controller_.get(), absl::StrCat("worker_", i),
                         src_workers_[i]->port, i + 1);
    }
    RegisterFakeWorker(dst_controller_.get(), "worker_0", dst_worker_.port,
                       1);

    ReshardService::Options dst_options;
    dst_options.port = 0;
    dst_options.delivery.mode = WorkerDelivery::Mode::kController;
    dst_options.delivery.controller = dst_controller_.get();
    dst_service_ = std::make_unique<ReshardService>(dst_options);
    ASSERT_TRUE(dst_service_->StartServer().ok());
    dst_address_ = absl::StrCat("127.0.0.1:", dst_service_->port());

    ReshardService::Options src_options;
    src_options.port = 0;
    src_options.delivery.mode = WorkerDelivery::Mode::kController;
    src_options.delivery.controller = src_controller_.get();
    src_service_ = std::make_unique<ReshardService>(src_options);

    RegisterUnits(num_src);
  }

  void RegisterUnits(int num_src) {
    // Source units register on the SOURCE service; the destination unit
    // registers on the DESTINATION service only (per-engine directories).
    for (int rank = 0; rank < num_src; ++rank) {
      rpcpb::ControlRequest req;
      req.set_command(rpcpb::ControlRequest::COMMAND_REGISTER_WORK_UNIT);
      auto* reg = req.mutable_register_work_unit_request();
      *reg->mutable_unit() = RaidenIdToProto(Unit(rank));
      reg->add_shards(absl::StrCat("10.0.0.1:", 9000 + rank));
      reg->set_control_plane_rpc_address(
          absl::StrCat("10.0.0.1:", 9100 + 2 * rank));
      *reg->add_pools() = MakePool("fa", 1024, 1024, 16);
      reg->set_layout_fingerprint("fp1");
      reg->set_page_tokens(512);
      reg->set_transfer_parallelism(num_src);
      reg->set_transfer_rank(rank);
      rpcpb::ControlResponse resp;
      resp.ParseFromString(src_service_->HandleFrame(req.SerializeAsString()));
      ASSERT_TRUE(resp.success()) << resp.message();
    }
    rpcpb::ControlRequest req;
    req.set_command(rpcpb::ControlRequest::COMMAND_REGISTER_WORK_UNIT);
    auto* reg = req.mutable_register_work_unit_request();
    *reg->mutable_unit() = RaidenIdToProto(DstUnit());
    reg->add_shards("10.0.0.2:9400");
    reg->set_control_plane_rpc_address("10.0.0.2:9600");
    *reg->add_pools() = MakePool("fa", 1024, 1024, 16);
    reg->set_layout_fingerprint("fp1");
    reg->set_page_tokens(4096);
    reg->set_transfer_parallelism(num_src);
    reg->set_transfer_rank(0);
    rpcpb::ControlResponse resp;
    resp.ParseFromString(dst_service_->HandleFrame(req.SerializeAsString()));
    ASSERT_TRUE(resp.success()) << resp.message();
  }

  void RegisterSpans(int rank, const std::string& req_id, int64_t uuid,
                     int64_t src_block, int64_t dst_index, int64_t dst_offset,
                     int64_t size) {
    rpcpb::ControllerRequest req;
    req.set_command(
        rpcpb::ControllerRequest::COMMAND_REGISTER_REQUEST_BLOCKS);
    auto* block_req = req.mutable_register_request_blocks_request();
    block_req->set_req_id(req_id);
    block_req->set_uuid(uuid);
    *block_req->mutable_unit() = RaidenIdToProto(Unit(rank));
    block_req->add_block_ids(src_block);
    auto* entry = block_req->add_pool_spans();
    entry->set_tag("fa");
    entry->add_block_ids(src_block);
    auto* span = entry->add_spans();
    span->set_src_block_ordinal(0);
    span->set_src_offset_bytes(0);
    span->set_dst_block_index(dst_index);
    span->set_dst_offset_bytes(dst_offset);
    span->set_size_bytes(size);
    span->set_count(1);
    entry->set_declared_bytes(size);
    rpcpb::ControllerResponse resp;
    resp.ParseFromString(src_service_->HandleFrame(req.SerializeAsString()));
    ASSERT_TRUE(resp.success()) << resp.message();
  }

  rpcpb::ControllerResponse Coordinate(const std::string& req_id,
                                       int64_t uuid, int num_src,
                                       std::vector<int64_t> dst_blocks) {
    rpcpb::ControllerRequest req;
    req.set_command(rpcpb::ControllerRequest::COMMAND_COORDINATE_TRANSFER);
    auto* coord = req.mutable_coordinate_transfer_request();
    for (int rank = 0; rank < num_src; ++rank) {
      *coord->add_src_units() = RaidenIdToProto(Unit(rank));
    }
    *coord->add_dst_units() = RaidenIdToProto(DstUnit());
    coord->set_uuid(uuid);
    coord->set_is_sender(true);
    coord->set_dst_mem_type(rpcpb::MEMORY_TYPE_HBM);
    coord->set_use_block_chunks(true);
    coord->set_req_id(req_id);
    coord->set_dst_controller_address(dst_address_);
    for (int64_t block : dst_blocks) {
      coord->add_dst_device_block_ids(block);
    }
    coord->add_transfer_pool_tags("fa");
    rpcpb::ControllerResponse resp;
    resp.ParseFromString(src_service_->HandleFrame(req.SerializeAsString()));
    return resp;
  }

  std::vector<std::unique_ptr<FakeWorker>> src_workers_;
  FakeWorker dst_worker_;
  std::unique_ptr<tpu_raiden::controller::RaidenController> src_controller_;
  std::unique_ptr<tpu_raiden::controller::RaidenController> dst_controller_;
  std::unique_ptr<ReshardService> src_service_;
  std::unique_ptr<ReshardService> dst_service_;
  std::string dst_address_;
};

TEST_F(ReshardPhaseBTest, ZeroSidecarFlowArmsAndDispatchesViaControllers) {
  BuildTopology(/*num_src=*/2);
  RegisterSpans(0, "req-b", 71, /*src_block=*/3, /*dst_index=*/0,
                /*dst_offset=*/0, /*size=*/1024);
  RegisterSpans(1, "req-b", 71, /*src_block=*/5, /*dst_index=*/1,
                /*dst_offset=*/0, /*size=*/512);

  rpcpb::ControllerResponse resp = Coordinate("req-b", 71, 2, {7, 9});
  ASSERT_TRUE(resp.success()) << resp.message();

  // The destination's fake worker was armed exactly once, through the
  // relay and the destination's OWN controller.
  ASSERT_EQ(dst_worker_.manager.arm_requests.size(), 1u);
  const rpcpb::StartTransferRequest& arm =
      dst_worker_.manager.arm_requests[0];
  EXPECT_FALSE(arm.is_sender());
  EXPECT_EQ(arm.shard_push_schedules_size(), 2);
  ASSERT_EQ(arm.pool_groups_size(), 1);
  EXPECT_EQ(arm.pool_groups(0).expected_pushes(), 2);
  EXPECT_EQ(dst_worker_.manager.arm_ids[0],
            core::DeriveArmChipBlockIds(arm));

  // Each sender's fake worker received its own single-rank program.
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(src_workers_[i]->manager.push_requests.size(), 1u)
        << "sender " << i;
    const rpcpb::StartTransferRequest& push =
        src_workers_[i]->manager.push_requests[0];
    EXPECT_TRUE(push.is_sender());
    EXPECT_EQ(push.shard_push_schedules_size(), 1);
    EXPECT_EQ(src_workers_[i]->manager.push_ids[0],
              core::DeriveSenderSourceBlockIds(push));
    EXPECT_EQ(src_workers_[i]->manager.push_parallelism[0],
              push.parallelism());
  }
}

TEST_F(ReshardPhaseBTest, RelayArmRefusalAbandonsClaimAndSkipsSenders) {
  BuildTopology(/*num_src=*/2);
  dst_worker_.manager.arm_result =
      absl::InternalError("injected arm refusal");
  RegisterSpans(0, "req-c", 72, 3, 0, 0, 1024);
  RegisterSpans(1, "req-c", 72, 5, 1, 0, 512);

  rpcpb::ControllerResponse resp = Coordinate("req-c", 72, 2, {7, 9});
  EXPECT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("injected arm refusal"));
  EXPECT_EQ(src_workers_[0]->manager.push_requests.size(), 0u);
  EXPECT_EQ(src_workers_[1]->manager.push_requests.size(), 0u);

  // The claim was abandoned: a fresh coordinate over re-registered spans
  // succeeds once the destination stops refusing.
  dst_worker_.manager.arm_result = absl::OkStatus();
  RegisterSpans(0, "req-c", 72, 3, 0, 0, 1024);
  RegisterSpans(1, "req-c", 72, 5, 1, 0, 512);
  rpcpb::ControllerResponse retry = Coordinate("req-c", 72, 2, {7, 9});
  ASSERT_TRUE(retry.success()) << retry.message();
}

TEST_F(ReshardPhaseBTest, MissingDestinationWorkerFailsBeforeSenders) {
  BuildTopology(/*num_src=*/1);
  // Rebuild the destination service around a controller with NO workers.
  auto empty_controller = MakeController(DstUnit());
  ReshardService::Options options;
  options.port = 0;
  options.delivery.mode = WorkerDelivery::Mode::kController;
  options.delivery.controller = empty_controller.get();
  auto lonely_service = std::make_unique<ReshardService>(options);
  ASSERT_TRUE(lonely_service->StartServer().ok());
  {
    rpcpb::ControlRequest req;
    req.set_command(rpcpb::ControlRequest::COMMAND_REGISTER_WORK_UNIT);
    auto* reg = req.mutable_register_work_unit_request();
    *reg->mutable_unit() = RaidenIdToProto(DstUnit());
    reg->add_shards("10.0.0.2:9400");
    reg->set_control_plane_rpc_address("10.0.0.2:9600");
    *reg->add_pools() = MakePool("fa", 1024, 1024, 16);
    reg->set_layout_fingerprint("fp1");
    reg->set_page_tokens(4096);
    reg->set_transfer_parallelism(1);
    reg->set_transfer_rank(0);
    rpcpb::ControlResponse resp;
    resp.ParseFromString(
        lonely_service->HandleFrame(req.SerializeAsString()));
    ASSERT_TRUE(resp.success()) << resp.message();
  }
  dst_address_ = absl::StrCat("127.0.0.1:", lonely_service->port());

  RegisterSpans(0, "req-d", 73, 3, 0, 0, 1024);
  rpcpb::ControllerResponse resp = Coordinate("req-d", 73, 1, {7});
  EXPECT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("not registered"));
  EXPECT_EQ(src_workers_[0]->manager.push_requests.size(), 0u);
}

TEST_F(ReshardPhaseBTest, FramedModeDestinationRejectsRelay) {
  BuildTopology(/*num_src=*/1);
  // A phase-A (framed) destination service must fail the relay closed.
  ReshardService::Options options;
  options.port = 0;
  auto framed_service = std::make_unique<ReshardService>(options);
  ASSERT_TRUE(framed_service->StartServer().ok());

  rpcpb::ControlRequest arm;
  arm.set_command(rpcpb::ControlRequest::COMMAND_START_TRANSFER);
  auto* start = arm.mutable_start_transfer_request();
  *start->add_dst_units() = RaidenIdToProto(DstUnit());
  start->set_is_sender(false);
  start->add_transfer_pool_indices(0);
  SocketFramedTransport transport;
  auto response =
      transport.Call(absl::StrCat("127.0.0.1:", framed_service->port()),
                     arm.SerializeAsString(), absl::Seconds(5));
  ASSERT_TRUE(response.ok()) << response.status();
  rpcpb::ControlResponse resp;
  ASSERT_TRUE(resp.ParseFromString(*response));
  EXPECT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("not served on this surface"));
}

}  // namespace
}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden
