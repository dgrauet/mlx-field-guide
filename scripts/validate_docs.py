#!/usr/bin/env python3
"""Validate the MLX Field Guide's markdown content.

Runs four checks and reports any failures, then exits non-zero if any found.
Designed to be called as a pre-commit hook from the repo root.

Checks:
  1. Broken internal links  - every relative [...](path) target must exist.
  2. Missing glossary anchors - every [...](glossary.md#slug) must resolve to an
     <a id="slug"></a> defined in glossary.md.
  3. Nested markdown links  - no [ ... [text](url) ... ](url) constructions.
  4. Code-block corruption  - no glossary.md# reference may appear inside a
     fenced code block.

Scope: all .md files under the repo root, excluding docs/ (plan/spec scratchpad)
and any .md files inside vendor directories.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

INCLUDE_ROOTS = ("01-foundations", "02-ecosystem", "03-contributing")
TOP_LEVEL = ("README.md", "glossary.md")
EXCLUDE_SEGMENTS = ("docs/superpowers",)


def gather_files() -> list[Path]:
    files: list[Path] = []
    for name in TOP_LEVEL:
        p = REPO / name
        if p.exists():
            files.append(p)
    for root in INCLUDE_ROOTS:
        base = REPO / root
        if not base.is_dir():
            continue
        for p in base.rglob("*.md"):
            if any(seg in str(p) for seg in EXCLUDE_SEGMENTS):
                continue
            files.append(p)
    return sorted(files)


def load_glossary_anchors() -> set[str]:
    glossary = REPO / "glossary.md"
    if not glossary.exists():
        return set()
    return set(re.findall(r'<a id="([^"]+)"', glossary.read_text()))


LINK_RE = re.compile(r"\]\(([^)]+)\)")
# Matches [ outer_prefix [inner_text](inner_url) outer_suffix ](outer_url)
# i.e. a markdown link whose display text contains another complete link.
NESTED_RE = re.compile(r"\[[^\]]*\[[^\]]+\]\([^)]+\)[^\]]*\]\([^)]+\)")
FENCE_RE = re.compile(r"```.*?```", flags=re.DOTALL)


def check_broken_internal_links(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for f in files:
        text = f.read_text()
        for m in LINK_RE.finditer(text):
            href = m.group(1).split()[0]
            if href.startswith(("http://", "https://", "mailto:")):
                continue
            path_part = href.split("#", 1)[0]
            if not path_part:
                continue
            target = (f.parent / path_part).resolve()
            if not target.exists():
                errors.append(f"{f.relative_to(REPO)}: broken link -> {href}")
    return errors


def check_missing_glossary_anchors(files: list[Path], anchors: set[str]) -> list[str]:
    errors: list[str] = []
    slug_re = re.compile(r"glossary\.md#([a-z0-9-]+)")
    for f in files:
        if f.name == "glossary.md":
            continue
        for m in slug_re.finditer(f.read_text()):
            slug = m.group(1)
            if slug not in anchors:
                errors.append(f"{f.relative_to(REPO)}: glossary.md#{slug} (anchor not defined)")
    return errors


def check_nested_links(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for f in files:
        for i, line in enumerate(f.read_text().splitlines(), start=1):
            for m in NESTED_RE.finditer(line):
                snippet = m.group(0)
                if len(snippet) > 100:
                    snippet = snippet[:97] + "..."
                errors.append(f"{f.relative_to(REPO)}:{i}: nested link -> {snippet}")
    return errors


def check_code_block_corruption(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for f in files:
        text = f.read_text()
        for m in FENCE_RE.finditer(text):
            if "glossary.md#" in m.group(0):
                line = text[: m.start()].count("\n") + 1
                errors.append(f"{f.relative_to(REPO)}:{line}: glossary link inside fenced code block")
                break
    return errors


def main() -> int:
    files = gather_files()
    if not files:
        print("validate_docs: no markdown files found -- is this the right repo?", file=sys.stderr)
        return 2

    anchors = load_glossary_anchors()

    sections = [
        ("Broken internal links", check_broken_internal_links(files)),
        ("Missing glossary anchors", check_missing_glossary_anchors(files, anchors)),
        ("Nested markdown links", check_nested_links(files)),
        ("Glossary links inside code blocks", check_code_block_corruption(files)),
    ]

    total = sum(len(errs) for _, errs in sections)
    if total == 0:
        print(f"validate_docs: OK ({len(files)} files, {len(anchors)} glossary anchors)")
        return 0

    print(f"validate_docs: FAIL -- {total} issue(s) across {len(files)} files\n", file=sys.stderr)
    for title, errs in sections:
        if not errs:
            continue
        print(f"## {title} ({len(errs)})", file=sys.stderr)
        for e in errs:
            print(f"  {e}", file=sys.stderr)
        print("", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
