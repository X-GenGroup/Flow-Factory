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

"""Search Flow-Factory's indexed historical fix patterns."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import yaml

TOKEN_PATTERN = re.compile(r"[a-z0-9_+-]+")
FIELD_PATTERN = re.compile(r"^- \*\*(?P<name>[^*]+)\*\*: (?P<value>.*)$", re.MULTILINE)


def _tokens(values: Iterable[str]) -> List[str]:
    """Return normalized search tokens while dropping low-value path words."""
    ignored = {"src", "flow_factory", "py", "tests", "test", "md"}
    return [
        token
        for value in values
        for token in TOKEN_PATTERN.findall(value.lower())
        if len(token) > 2 and token not in ignored
    ]


def _load_index(index_path: Path) -> List[Dict[str, Any]]:
    """Load the machine-readable fix inventory."""
    data = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("patterns"), list):
        raise TypeError(f"expected patterns list in {index_path}")
    return data["patterns"]


def _parse_domain(path: Path) -> Dict[str, str]:
    """Map normalized Markdown heading anchors to complete fix cards."""
    text = path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^### (?P<title>.+)$", text))
    cards: Dict[str, str] = {}
    for index, match in enumerate(matches):
        stop = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else text.find("## Cross-refs", match.end())
        )
        title = match.group("title")
        anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:96].rstrip("-")
        cards[anchor] = text[match.start() : stop].strip()
    return cards


def search_patterns(
    fixes_dir: Path,
    *,
    query: str = "",
    paths: Iterable[str] = (),
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Return the highest-scoring fix patterns for a query and changed paths."""
    path_values = list(paths)
    tokens = _tokens([query, *path_values])
    phrase = query.strip().lower()
    cards_by_file: Dict[str, Dict[str, str]] = {}
    results = []

    for item in _load_index(fixes_dir / "index.yaml"):
        filename = item["file"]
        if filename not in cards_by_file:
            cards_by_file[filename] = _parse_domain(fixes_dir / filename)
        card = cards_by_file[filename].get(item["anchor"], "")
        searchable = f"{item['title']} {item['domain']} {card}".lower()
        score = sum(
            3 if token in item["title"].lower() else 1 for token in tokens if token in searchable
        )
        if phrase and phrase in searchable:
            score += 8
        for path in path_values:
            if item["domain"].replace("-", "_") in path.lower():
                score += 3
        if score <= 0:
            continue
        fields = {
            match.group("name"): match.group("value") for match in FIELD_PATTERN.finditer(card)
        }
        results.append(
            {
                "id": item["id"],
                "title": item["title"],
                "domain": item["domain"],
                "file": str(fixes_dir / filename),
                "score": score,
                "symptom": fields.get("Symptom", ""),
                "lesson": fields.get("Lesson", ""),
            }
        )

    results.sort(key=lambda result: (-result["score"], result["title"]))
    return results[:limit]


def main() -> int:
    """Search the repository fix index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="", help="Symptom, invariant, or owner")
    parser.add_argument("--paths", nargs="*", default=(), help="Changed repository paths")
    parser.add_argument("--limit", type=int, default=5, help="Maximum results")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()
    if not args.query and not args.paths:
        parser.error("provide --query or --paths")
    if args.limit < 1:
        parser.error("--limit must be positive")

    root = Path(__file__).resolve().parents[1]
    results = search_patterns(
        root / ".agents/knowledge/fixes",
        query=args.query,
        paths=args.paths,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for result in results:
            print(f"[{result['domain']}] {result['title']}")
            print(f"  {result['file']}#{result['id']}")
            if result["symptom"]:
                print(f"  Symptom: {result['symptom']}")
            if result["lesson"]:
                print(f"  Lesson: {result['lesson']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
