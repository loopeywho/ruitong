"""OpenAI-compatible HTTP backend for vLLM and vllm-ascend.

One class, instantiated twice (name="cuda" / name="ascend") with different
base URLs.  Both backends speak the same OpenAI-compatible API; no branching
on backend name.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from ..errors import BackendError, BackendUnavailable
from ..schemas import ChatChunk, ChatRequest, ChatResponse, HealthStatus, ModelInfo


class VllmHttpBackend:
    """HTTP backend for vLLM-compatible servers (CUDA and Ascend)."""

    name: str

    def __init__(
        self, name: str, base_url: str, timeout: float = 60.0
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _new_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout)

    async def _request(
        self, method: str, url: str, json_data: dict | None = None
    ) -> httpx.Response:
        """Send HTTP request with one bounded retry on connection errors."""
        errors: list[Exception] = []
        for _attempt in range(2):  # one retry
            try:
                async with self._new_client() as client:
                    response = await client.request(method, url, json=json_data)
                    if response.is_error:
                        exc = httpx.HTTPStatusError(
                            f"HTTP {response.status_code}: "
                            f"{response.text[:200]}",
                            request=response.request,
                            response=response,
                        )
                        raise BackendError(
                            self.name,
                            message=exc.args[0],
                        ) from exc
                    return response
            except (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.NetworkError,
            ) as exc:
                errors.append(exc)
                continue  # retry
        raise BackendUnavailable(
            self.name, message=f"All {len(errors)} attempts failed"
        ) from errors[-1]

    async def health(self) -> HealthStatus:
        """Return health status; never raises — wraps errors."""
        try:
            url = f"{self.base_url}/v1/models"
            async with self._new_client() as client:
                resp = await client.get(url)
            if resp.status_code != 200:
                return HealthStatus(
                    healthy=False, backend=self.name, models_served=0
                )
            models = resp.json()
            return HealthStatus(
                healthy=True, backend=self.name, models_served=len(models)
            )
        except Exception:
            return HealthStatus(healthy=False, backend=self.name, models_served=0)

    async def list_models(self) -> list[ModelInfo]:
        url = f"{self.base_url}/v1/models"
        response = await self._request("GET", url)
        try:
            ids = response.json()
        except Exception as exc:
            raise BackendError(
                self.name, cause=exc, message="Malformed response body"
            ) from exc
        if not isinstance(ids, list):
            ids = [ids]
        return [
            ModelInfo(id=mid if isinstance(mid, str) else mid.get("id", ""), backend=self.name)
            for mid in ids
        ]

    async def chat(self, req: ChatRequest) -> ChatResponse:
        url = f"{self.base_url}/v1/chat/completions"
        response = await self._request(
            "POST", url, json_data=req.model_dump()
        )
        try:
            body = response.json()
        except Exception as exc:
            raise BackendError(
                self.name, cause=exc, message="Malformed response body"
            ) from exc
        if not isinstance(body, dict) or not body.get("choices"):
            raise BackendError(
                self.name, message="Empty response choices"
            ) from None
        body["backend"] = self.name
        return ChatResponse.model_validate(body)

    # NOTE: deliberately `async def` here.
    #
    # An async generator *is* an AsyncIterator, so this satisfies the
    # Backend.stream() protocol method.
    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Yield streaming chunks for a chat request.

        Parses SSE (Server-Sent Events) from the vLLM response body.
        """
        url = f"{self.base_url}/v1/chat/completions"
        response = await self._request(
            "POST",
            url,
            json_data={**req.model_dump(), "stream": True},
        )
        body = response.text
        chunks_yielded = 0
        for line in body.split("\n"):
            if line.startswith("data: "):
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                if data:
                    try:
                        chunk_data = json.loads(data)
                        if chunk_data.get("choices"):
                            chunk_data["backend"] = self.name
                            yield ChatChunk.model_validate(chunk_data)
                            chunks_yielded += 1
                    except json.JSONDecodeError:
                        pass
        if chunks_yielded == 0:
            raise BackendError(
                self.name, message="Empty response (no chunks)"
            ) from None
