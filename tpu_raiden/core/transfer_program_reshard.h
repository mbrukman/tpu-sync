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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_TRANSFER_PROGRAM_RESHARD_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_TRANSFER_PROGRAM_RESHARD_H_

// Pool-reshard <-> TransferProgram bijection and the phase-B program
// validator (reshard migration plan §4.6).
//
// Design invariant: the phase-B worker door executes through the SAME
// manager operations the framed listener drives (PoolReshardPush /
// PoolReshardRegisterRecv, both consuming a whole StartTransferRequest).
// CompileStartTransfer and LowerToStartTransfer are exact inverses for the
// pool-reshard class, so the manager sees byte-identical inputs under either
// door and every receiver-side proof (D3) is untouched. Gate GB1 replays the
// live G1 corpus through compile-then-lower and byte-compares under
// deterministic serialization.

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "tpu_raiden/proto/transfer_program.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace core {

// The lowering classes phase B implements. Everything else is rejected by
// NormalizeReshardProgram (AwaitInline / Barrier contracts: Unimplemented,
// dormant until phase C; malformed programs: InvalidArgument).
enum class ReshardLoweringClass {
  kPoolReshardSender,
  kPoolReshardArm,
};

// B2 — the normalizing validator. Admits a program into exactly one lowering
// class or rejects it. Dispatch keys on the completion contract and role,
// never on envelope.kind (kind is checked only for consistency).
absl::StatusOr<ReshardLoweringClass> NormalizeReshardProgram(
    const proto::TransferProgramRequest& request);

// B5 (forward) — lossless pool-reshard StartTransferRequest -> program.
// Rejects non-pool requests (legacy dense plans stay on the framed door).
absl::StatusOr<proto::TransferProgramRequest> CompileStartTransfer(
    const rpc::StartTransferRequest& request);

// Worker-door lowering (inverse of CompileStartTransfer). Callers must have
// run NormalizeReshardProgram first; this reconstructs the byte-identical
// StartTransferRequest the framed door would have delivered.
absl::StatusOr<rpc::StartTransferRequest> LowerToStartTransfer(
    const proto::TransferProgramRequest& request);

// Listener-parity block-id derivations (kv_cache_listener.cc sender/receiver
// paths): the sender's local block list is the sorted-unique union of its
// schedule entries' source blocks; the receiver's flat destination list is
// the pool groups' dst_device_block_ids concatenation.
std::vector<int64_t> DeriveSenderSourceBlockIds(
    const rpc::StartTransferRequest& request);
std::vector<int64_t> DeriveArmChipBlockIds(
    const rpc::StartTransferRequest& request);

}  // namespace core
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_TRANSFER_PROGRAM_RESHARD_H_
