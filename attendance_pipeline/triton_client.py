from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TritonModelInput:
    name: str
    data: np.ndarray
    datatype: str = "FP32"


class TritonInferenceClient:
    def __init__(self, url: str, timeout_sec: float = 5.0, enabled: bool = True):
        self.url = url
        self.timeout_sec = timeout_sec
        self.enabled = enabled
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if not self.enabled:
            raise RuntimeError("Triton is disabled")
        if self._client is None:
            import tritonclient.http as httpclient

            self._client = httpclient.InferenceServerClient(url=self.url, connection_timeout=self.timeout_sec)
        return self._client

    def infer(self, model_name: str, inputs: list[TritonModelInput], output_names: list[str]) -> list[np.ndarray]:
        import tritonclient.http as httpclient

        triton_inputs = []
        for item in inputs:
            infer_input = httpclient.InferInput(item.name, item.data.shape, item.datatype)
            infer_input.set_data_from_numpy(item.data)
            triton_inputs.append(infer_input)

        outputs = [httpclient.InferRequestedOutput(name) for name in output_names]
        result = self.client.infer(model_name, triton_inputs, outputs=outputs, request_timeout=self.timeout_sec)
        return [result.as_numpy(name) for name in output_names]

