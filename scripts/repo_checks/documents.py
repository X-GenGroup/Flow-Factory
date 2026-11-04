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

"""Local Markdown references and the roadmap's existing work-item graph."""

import posixpath
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

from .core import Repository

MARKDOWN = MarkdownIt("commonmark").enable("table")
ITEM = re.compile(r"^(P[0-9]{2}[a-z]?):\s")
STATUSES = {"planned", "in_progress", "blocked", "done", "deferred"}


@dataclass
class Document:
    anchors: set[str] = field(default_factory=set)
    links: list[tuple[int, str]] = field(default_factory=list)
    headings: list[tuple[int, str]] = field(default_factory=list)


class HTMLReferences(HTMLParser):
    def __init__(self, document: Document, line: int):
        super().__init__()
        self.document = document
        self.line = line

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if value is None:
                continue
            if key == "id" or (tag == "a" and key == "name"):
                self.document.anchors.add(value)
            if key in ("href", "src"):
                self.document.links.append((self.line + self.getpos()[0] - 1, value))


def parse_document(text: str) -> Document:
    result = Document()
    slugs: set[str] = set()
    heading = False
    for token in MARKDOWN.parse(text):
        line = token.map[0] + 1 if token.map else 1
        if token.type == "heading_open":
            heading = True
        elif token.type == "inline":
            children = token.children or []
            if heading:
                label = "".join(
                    c.content for c in children if c.type in ("text", "code_inline", "image")
                )
                slug = re.sub(r"[^\w\- ]", "", label.lower()).replace(" ", "-")
                anchor, suffix = slug, 0
                while anchor in slugs:
                    suffix += 1
                    anchor = f"{slug}-{suffix}"
                slugs.add(anchor)
                result.anchors.add(anchor)
                result.headings.append((line, label))
                heading = False
            for child in children:
                if child.type in ("link_open", "image"):
                    result.links.append((line, child.attrGet("href") or child.attrGet("src") or ""))
                if child.type == "html_inline":
                    HTMLReferences(result, line).feed(child.content)
        elif token.type == "html_block":
            HTMLReferences(result, line).feed(token.content)
    return result


def owned_document(path: str) -> bool:
    return path.endswith(".md") and (
        "/" not in path
        or path.startswith(("guidance/", "roadmap/", ".agents/", "examples/"))
        or path.endswith("/AGENTS.md")
    )


def local_target(path: str, href: str) -> tuple[str, str] | None:
    url = urlsplit(href)
    if url.scheme or url.netloc:
        return None
    target = unquote(url.path)
    if not target:
        target = path
    elif target.startswith("/"):
        target = target.lstrip("/")
    else:
        target = posixpath.join(posixpath.dirname(path), target)
    return posixpath.normpath(target), unquote(url.fragment)


def check_documents(repo: Repository) -> dict[str, Document]:
    documents = {
        path: parse_document(repo.read(path))
        for path in sorted(repo.paths)
        if owned_document(path) and repo.exists(path)
    }
    for path, doc in list(documents.items()):
        for line, href in doc.links:
            try:
                target = local_target(path, href)
                if target is None:
                    continue
                destination, fragment = target
                if destination == ".." or destination.startswith("../"):
                    raise ValueError(f"link escapes repository: {href!r}")
                if not repo.exists(destination):
                    if not repo.directory(destination):
                        raise ValueError(
                            f"missing local target {href!r} (resolved to {destination})"
                        )
                    if not fragment:
                        continue
                    destination = posixpath.join(destination, "README.md")
                    if not repo.exists(destination):
                        raise ValueError(f"directory anchor requires a tracked README.md: {href!r}")
                if fragment and destination.endswith(".md"):
                    if destination not in documents:
                        documents[destination] = parse_document(repo.read(destination))
                    if fragment not in documents[destination].anchors:
                        raise ValueError(f"missing anchor {fragment!r} in {destination}")
            except ValueError as error:
                repo.error(path, line, "docs-link", str(error))
    return documents


def check_roadmap(repo: Repository, documents: dict[str, Document]) -> None:
    items: dict[str, tuple[str, int, str]] = {}
    for path, doc in documents.items():
        if not path.startswith("roadmap/milestones/"):
            continue
        lines = repo.read(path).splitlines()
        headings = [
            (line, ITEM.match(label).group(1)) for line, label in doc.headings if ITEM.match(label)
        ]
        for index, (line, name) in enumerate(headings):
            end = headings[index + 1][0] - 1 if index + 1 < len(headings) else len(lines)
            if name in items:
                repo.error(
                    path,
                    line,
                    "roadmap-id",
                    f"duplicate work-item ID {name}; first in {items[name][0]}",
                )
            items[name] = (path, line, "\n".join(lines[line:end]))
            if name.lower() not in doc.anchors:
                repo.error(path, line, "roadmap-id", f"{name} needs stable anchor {name.lower()!r}")
    graph: dict[str, set[str]] = {}
    for name, (path, line, body) in items.items():
        status = re.findall(r"^- \*\*Status:\*\*\s*(\S+)", body, re.MULTILINE)
        if len(status) != 1 or status[0] not in STATUSES:
            repo.error(
                path,
                line,
                "roadmap-status",
                f"{name}: expected one documented Status, received {status}",
            )
        dependencies = re.findall(r"^- \*\*Depends on:\*\*\s*(.+)", body, re.MULTILINE)
        graph[name] = set()
        if len(dependencies) != 1:
            repo.error(path, line, "roadmap-dependency", f"{name}: expected one Depends on field")
            continue
        # Linked IDs are executable dependencies; prose after them can qualify release gates.
        linked = re.findall(r"\[(P[0-9]{2}[a-z]?)\]\(([^)]+)\)", dependencies[0])
        if not linked and dependencies[0].strip() != "none":
            repo.error(
                path,
                line,
                "roadmap-dependency",
                f"{name}: dependencies must be linked IDs or 'none'",
            )
        for dependency, href in linked:
            if dependency not in items:
                repo.error(
                    path, line, "roadmap-dependency", f"{name}: unknown dependency {dependency}"
                )
            elif local_target(path, href) != (items[dependency][0], dependency.lower()):
                repo.error(
                    path,
                    line,
                    "roadmap-dependency",
                    f"{name}: link for {dependency} points to the wrong work item",
                )
            else:
                graph[name].add(dependency)
    visited: set[str] = set()

    def visit(name: str, stack: tuple[str, ...]) -> None:
        if name in stack:
            path, line, _ = items[name]
            repo.error(path, line, "roadmap-cycle", " -> ".join((*stack, name)))
            return
        if name in visited:
            return
        for dependency in sorted(graph[name]):
            visit(dependency, (*stack, name))
        visited.add(name)

    for name in sorted(graph):
        visit(name, ())
