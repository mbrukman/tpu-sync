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
"""Engine-hosted reshard store (phase B of the reshard C++ migration).

Zero-sidecar hosting: the serving deployment's transfer-rank-0 worker
constructs one ReshardStore per engine. It owns two surfaces for the
engine:

- the framed reshard service (work-unit directory, request-block registry,
  planner, coordinator) in controller-delivery mode — the same wire the
  external controller served, now in-process;
- the dispatch RaidenController the engine's local workers register with;
  the coordinator submits transfer programs over its persistent
  WorkerService channels, and the destination-side receiver-arm relay
  dispatches through it.

Keep the instance alive for the process lifetime; dropping it stops both
surfaces.
"""

from tpu_raiden.api.torch import torch_tpu_common_loader

torch_tpu_common_loader.load_torch_tpu_common()

# pylint: disable=g-import-not-at-top
from tpu_raiden.api.torch import torch_abi

_impl = torch_abi.load_extension(
    "tpu_raiden.frameworks.torch",
    "_tpu_raiden_torch",
)
# pylint: enable=g-import-not-at-top


class ReshardStore:
  """Wrapper around the compiled engine-hosted reshard store."""

  def __init__(
      self,
      reshard_port: int,
      dispatch_bind_address: str,
      request_registry_ttl_s: float = 600.0,
  ):
    """Binds the reshard service and the dispatch controller.

    Args:
      reshard_port: port for the framed reshard surface (the address peers
        and the local facade dial; advertised via the connector handoff).
      dispatch_bind_address: host:port the dispatch controller binds for
        worker RegisterWorker calls.
      request_registry_ttl_s: request-block registry TTL, matching the
        external controller's --request-registry-ttl-s.
    """
    self._impl = _impl.ReshardStore(
        reshard_port=reshard_port,
        dispatch_bind_address=dispatch_bind_address,
        request_registry_ttl_s=request_registry_ttl_s,
    )

  @property
  def reshard_port(self) -> int:
    """The actually-bound reshard service port."""
    return int(self._impl.reshard_port)
