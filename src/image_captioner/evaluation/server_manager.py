"""Manages the lifecycle of a local llama-server subprocess per candidate model."""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import requests

from image_captioner.evaluation.config import Candidate

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_llama_server.sh"


class ServerStartupError(Exception):
    """Raised when llama-server does not become healthy within the timeout."""


@dataclass
class ServerHandle:
    """A running llama-server process plus the log file capturing its output."""

    process: subprocess.Popen
    log_file: IO[str]


def start_server(candidate: Candidate, port: int, log_path: Path) -> ServerHandle:
    args = ["bash", str(SCRIPT_PATH), str(candidate.model_path)]
    if candidate.mmproj_path is not None:
        args.append(str(candidate.mmproj_path))
    env = dict(os.environ)
    env["LLAMA_SERVER_PORT"] = str(port)
    log_file = open(log_path, "w")
    process = subprocess.Popen(args, env=env, stdout=log_file, stderr=log_file)
    return ServerHandle(process=process, log_file=log_file)


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


def stop_server(handle: ServerHandle, timeout: float = 10.0) -> None:
    try:
        handle.process.terminate()
        try:
            handle.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            handle.process.kill()
            handle.process.wait()
    finally:
        handle.log_file.close()
