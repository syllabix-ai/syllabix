#!/usr/bin/env bash
# Sync release-tag pins in the install docs to one version.
#
# Usage: scripts/sync-release-pins.sh <version>   (e.g. v0.1.1 or 0.1.1)
#
# The install docs point users at exact git tags (`--tag vX.Y.Z` for
# `cargo install --git`, `tag = "vX.Y.Z"` for git dependencies) plus the
# README version badge. Cargo has no "latest tag" syntax, so these pins are
# rewritten on every release: by hand with this script, or automatically by
# the sync-pins job in .github/workflows/release.yml, which opens a PR
# after a version tag is pushed.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${root}"

version="${1:-}"
version="${version#v}"
if ! [[ "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "usage: $0 <version>   (e.g. $0 v0.1.1)" >&2
    exit 1
fi

python3 - "${version}" <<'PY'
import pathlib
import re
import sys

version = sys.argv[1]

# (file, pattern): every listed file must contain at least one match, so a
# docs refactor that moves a snippet fails loudly instead of silently
# dropping it from coverage.
targets = [
    ("README.md", r"--tag v\d+\.\d+\.\d+"),
    ("docs/install.md", r"--tag v\d+\.\d+\.\d+"),
    ("docs/embed.md", r"--tag v\d+\.\d+\.\d+"),
    ("crates/syllabix/README.md", r"--tag v\d+\.\d+\.\d+"),
    ("docs/embed.md", r'tag = "v\d+\.\d+\.\d+"'),
    ("crates/syllabix-core/README.md", r'tag = "v\d+\.\d+\.\d+"'),
    ("README.md", r"badge/version-v\d+\.\d+\.\d+-"),
]

failed = False
for rel, pattern in targets:
    path = pathlib.Path(rel)
    try:
        text = path.read_text()
    except FileNotFoundError:
        print(f"missing file: {rel}", file=sys.stderr)
        failed = True
        continue
    new_text, n = re.subn(
        pattern,
        lambda m: re.sub(r"v\d+\.\d+\.\d+", f"v{version}", m.group(0)),
        text,
    )
    if n == 0:
        print(f"no pin found: {rel} ({pattern})", file=sys.stderr)
        failed = True
        continue
    if new_text != text:
        path.write_text(new_text)
    print(f"{rel}: {n} pin(s) now v{version}")

sys.exit(1 if failed else 0)
PY
