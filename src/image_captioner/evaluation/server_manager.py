"""Manages the lifecycle of a local llama-server subprocess per candidate model."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import requests

from image_captioner.evaluation.config import Candidate

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_llama_server.sh"


class ServerStartupError(Exception):
    """Raised when llama-server does not become healthy within the timeout."""


def start_server(candidate: Candidate, port: int) -> subprocess.Popen:
    args = ["bash", str(SCRIPT_PATH), str(candidate.model_path)]
    if candidate.mmproj_path is not None:
        args.append(str(candidate.mmproj_path))
    env = dict(os.environ)
    env["LLAMA_SERVER_PORT"] = str(port)
    return subprocess.Popen(args, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_health(port: int, timeout: float, poll_interval: float = 1.0) -> None:
    endpoint = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = requests.get(endpoint, timeout=2.0)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(poll_interval)
    raise ServerStartupError(
        f"llama-server did not become healthy within {timeout}s on port {port}"
    )


def stop_server(process: subprocess.Popen, timeout: float = 10.0) -> None:
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
