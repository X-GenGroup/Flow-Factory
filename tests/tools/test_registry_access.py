# Copyright 2026 Jayce-Ping
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

"""Keep upload authorization distinct from optional registry cleanup support."""

import importlib.util
import io
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "registry_access", ROOT / "docker/check_registry_access.py"
)
REGISTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REGISTRY)
UPLOAD = "https://ghcr.io/v2/example/project/blobs/uploads/opaque-id"


def response(status, body=b"", **headers):
    result = io.BytesIO(body)
    result.status = status
    result.headers = headers
    return result


@pytest.fixture
def registry_http(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "Example/Project")
    monkeypatch.setenv("GITHUB_ACTOR", "example")
    monkeypatch.setenv("GH_TOKEN", "test-only-token")
    mock = Mock()
    monkeypatch.setattr(REGISTRY, "urlopen", mock)
    return mock


@pytest.mark.parametrize("status", [204, 405])
@pytest.mark.parametrize("location", [UPLOAD, "/v2/example/project/blobs/uploads/opaque-id"])
def test_write_access_does_not_require_delete_support(registry_http, capsys, status, location):
    cleanup = response(204) if status == 204 else HTTPError(UPLOAD, status, "Not allowed", {}, None)
    registry_http.side_effect = [
        response(200, b'{"token": "test-registry-token"}'),
        response(202, Location=location),
        cleanup,
    ]
    REGISTRY.main()
    output = capsys.readouterr().out
    assert "GHCR write access verified" in output
    assert "test-only-token" not in output
    assert "test-registry-token" not in output
    upload, cancel = [call.args[0] for call in registry_http.call_args_list[1:]]
    assert upload.get_method() == "POST" and upload.data == b""
    assert cancel.get_method() == "DELETE" and cancel.full_url == UPLOAD
    if status == 405:
        assert "not supported" in output
        assert "cancelled" not in output


@pytest.mark.parametrize(
    "phase, status",
    [
        ("token exchange", 403),
        ("start empty upload", 403),
        ("cancel empty upload", 403),
        ("token exchange", 405),
        ("start empty upload", 405),
    ],
)
def test_other_registry_failures_remain_fatal(registry_http, phase, status):
    responses = [
        response(200, b'{"token": "test-registry-token"}'),
        response(202, Location=UPLOAD),
    ]
    index = ["token exchange", "start empty upload", "cancel empty upload"].index(phase)
    registry_http.side_effect = responses[:index] + [
        HTTPError(UPLOAD, status, "Rejected", {}, None)
    ]
    with pytest.raises(SystemExit, match=f"GHCR {phase} failed \\(HTTP {status}\\)"):
        REGISTRY.main()
    assert registry_http.call_count == index + 1


def test_never_sends_registry_token_to_a_foreign_upload_host(registry_http):
    registry_http.side_effect = [
        response(200, b'{"token": "test-registry-token"}'),
        response(202, Location="https://untrusted.example/upload"),
    ]
    with pytest.raises(RuntimeError, match="unexpected upload location"):
        REGISTRY.main()
    assert registry_http.call_count == 2
