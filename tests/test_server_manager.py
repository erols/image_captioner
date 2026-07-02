import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from image_captioner.evaluation.config import Candidate
from image_captioner.evaluation.server_manager import (
    ServerStartupError,
    start_server,
    stop_server,
    wait_for_health,
)


def test_start_server_includes_mmproj_and_port_env() -> None:
    candidate = Candidate(
        name="qwen3-vl-8b",
        model_path=Path("/models/qwen3-vl-8b.gguf"),
        mmproj_path=Path("/models/qwen3-vl-8b-mmproj.gguf"),
    )

    with patch("image_captioner.evaluation.server_manager.subprocess.Popen") as mock_popen:
        start_server(candidate, port=8091)

    args, kwargs = mock_popen.call_args
    command = args[0]
    assert command[-2:] == ["/models/qwen3-vl-8b.gguf", "/models/qwen3-vl-8b-mmproj.gguf"]
    assert kwargs["env"]["LLAMA_SERVER_PORT"] == "8091"


def test_start_server_omits_mmproj_when_none() -> None:
    candidate = Candidate(name="text-only", model_path=Path("/models/text-only.gguf"))

    with patch("image_captioner.evaluation.server_manager.subprocess.Popen") as mock_popen:
        start_server(candidate, port=8092)

    args, _ = mock_popen.call_args
    command = args[0]
    assert command[-1] == "/models/text-only.gguf"


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

    stop_server(process)

    process.terminate.assert_called_once()
    process.wait.assert_called_once()
    process.kill.assert_not_called()


def test_stop_server_kills_when_terminate_times_out() -> None:
    process = MagicMock(spec=subprocess.Popen)
    process.wait.side_effect = [subprocess.TimeoutExpired(cmd="x", timeout=10.0), 0]

    stop_server(process)

    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2
