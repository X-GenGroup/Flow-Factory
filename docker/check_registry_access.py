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

"""Check GHCR write access by opening and cancelling an empty blob upload."""

import base64
import json
import os
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


def main() -> None:
    repository = os.environ["GITHUB_REPOSITORY"].lower()
    credentials = f"{os.environ['GITHUB_ACTOR']}:{os.environ['GH_TOKEN']}"
    basic = base64.b64encode(credentials.encode()).decode()
    query = urlencode({"service": "ghcr.io", "scope": f"repository:{repository}:pull,push"})
    request = Request(f"https://ghcr.io/token?{query}", headers={"Authorization": f"Basic {basic}"})
    try:
        with urlopen(request, timeout=30) as response:
            token = json.load(response)["token"]
        headers = {"Authorization": f"Bearer {token}"}
        request = Request(
            f"https://ghcr.io/v2/{repository}/blobs/uploads/",
            data=b"",
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            if response.status != 202:
                raise RuntimeError(f"Expected upload status 202, received {response.status}")
            location = response.headers["Location"]
        parsed = urlparse(location)
        if parsed.scheme != "https" or parsed.netloc != "ghcr.io":
            raise RuntimeError("GHCR returned an unexpected upload location")
        with urlopen(Request(location, headers=headers, method="DELETE"), timeout=30) as response:
            if response.status != 204:
                raise RuntimeError(f"Expected cancellation status 204, received {response.status}")
    except HTTPError as error:
        raise SystemExit(
            f"GHCR write preflight failed (HTTP {error.code}). Grant this repository Write access "
            "under the existing package's Manage Actions access before rebuilding."
        ) from None
    print("GHCR write access verified; empty upload cancelled, no image tag created.")


if __name__ == "__main__":
    main()
