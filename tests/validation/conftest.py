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

"""Local-only HTTP reward fixture shared by selected protocol checks."""

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from PIL import Image


@pytest.fixture
def reward_service():
    """Serve deterministic image-dependent scores without a deployed reward model."""
    batches = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.respond(200 if self.path == "/health" else 404, {"status": "ok"})

        def do_POST(self):
            if self.path != "/compute":
                self.respond(404, {"error": "unknown endpoint"})
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompts, images = payload["prompt"], payload["image"]
                if not prompts or len(prompts) != len(images):
                    raise ValueError("expected one image per prompt")
                scores = []
                for image in images:
                    with Image.open(io.BytesIO(base64.b64decode(image))) as decoded:
                        scores.append(decoded.convert("RGB").getpixel((0, 0))[0] / 255.0)
                batches.append(list(prompts))
                self.respond(200, {"rewards": scores})
            except (KeyError, TypeError, ValueError) as error:
                self.respond(400, {"error": str(error)})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_port}", batches=batches)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)
        assert not thread.is_alive()
