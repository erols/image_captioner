import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from image_captioner.evaluation.config import Candidate
from image_captioner.evaluation.server_manager import (
    ServerHandle,
    ServerStartupError,
    start_server,
    stop_server,
    wait_for_health,
)


def test_start_server_includes_mmproj_and_port_env(tmp_path: Path) -> None:
    candidate = Candidate(
        name="qwen3-vl-8b",
        model_path=Path("/models/qwen3-vl-8b.gguf"),
        mmproj_path=Path("/models/qwen3-vl-8b-mmproj.gguf"),
    )
    log_path = tmp_path / "qwen3-vl-8b-llama-server.log"

    with patch("image_captioner.evaluation.server_manager.subprocess.Popen") as mock_popen:
        handle = start_server(candidate, port=8091, log_path=log_path)

    args, kwargs = mock_popen.call_args
    command = args[0]
    assert command[-2:] == ["/models/qwen3-vl-8b.gguf", "/models/qwen3-vl-8b-mmproj.gguf"]
    assert kwargs["env"]["LLAMA_SERVER_PORT"] == "8091"
    assert isinstance(handle, ServerHandle)
    handle.log_file.close()


def test_start_server_omits_mmproj_when_none(tmp_path: Path) -> None:
    candidate = Candidate(name="text-only", model_path=Path("/models/text-only.gguf"))
    log_path = tmp_path / "text-only-llama-server.log"

    with patch("image_captioner.evaluation.server_manager.subprocess.Popen") as mock_popen:
        handle = start_server(candidate, port=8092, log_path=log_path)

    args, _ = mock_popen.call_args
    command = args[0]
    assert command[-1] == "/models/text-only.gguf"
    handle.log_file.close()


def test_start_server_redirects_stdout_and_stderr_to_log_file(tmp_path: Path) -> None:
    candidate = Candidate(name="text-only", model_path=Path("/models/text-only.gguf"))
    log_path = tmp_path / "text-only-llama-server.log"

    with patch("image_captioner.evaluation.server_manager.subprocess.Popen") as mock_popen:
        handle = start_server(candidate, port=8093, log_path=log_path)

    _, kwargs = mock_popen.call_args
    assert kwargs["stdout"] is handle.log_file
    assert kwargs["stderr"] is handle.log_file
    assert kwargs["stdout"] is not subprocess.DEVNULL
    assert log_path.exists()
    handle.log_file.close()


def test_wait_for_health_returns_when_server_responds_ok() -> None:
    fake_response = MagicMock()
    fake_response.status_code = 200

    with patch(
        "image_captioner.evaluation.server_manager.requests.get", return_value=fake_response
    ):
        wait_for_health(port=8091, timeout=5.0, poll_interval=0.01)


def test_wait_for_health_raises_after_timeout() -> None:
    with patch(
        "image_captioner.evaluation.server_manager.requests.get",
        side_effect=requests.ConnectionError("not up yet"),
    ):
        with pytest.raises(ServerStartupError):
            wait_for_health(port=8091, timeout=0.05, poll_interval=0.01)


def test_stop_server_terminates_and_waits() -> None:
    process = MagicMock(spec=subprocess.Popen)
    process.wait.return_value = 0
    log_file = MagicMock()
    handle = ServerHandle(process=process, log_file=log_file)

    stop_server(handle)

    process.terminate.assert_called_once()
    process.wait.assert_called_once()
    process.kill.assert_not_called()
    log_file.close.assert_called_once()


def test_stop_server_kills_when_terminate_times_out() -> None:
    process = MagicMock(spec=subprocess.Popen)
    process.wait.side_effect = [subprocess.TimeoutExpired(cmd="x", timeout=10.0), 0]
    log_file = MagicMock()
    handle = ServerHandle(process=process, log_file=log_file)

    stop_server(handle)

    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2
    log_file.close.assert_called_once()


def test_stop_server_closes_log_file_even_if_wait_raises() -> None:
    process = MagicMock(spec=subprocess.Popen)
    process.wait.side_effect = RuntimeError("boom")
    log_file = MagicMock()
    handle = ServerHandle(process=process, log_file=log_file)

    with pytest.raises(RuntimeError):
        stop_server(handle)

    log_file.close.assert_called_once()
