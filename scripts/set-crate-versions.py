#!/usr/bin/env python3
"""Write the release version into Cargo.toml and Cargo.lock for this job only.

Cargo.toml in git stays at 0.1.0. Maintainers never bump it by hand. At
release time the version comes from the git tag (vX.Y.Z → X.Y.Z). Both
crates-io.yml and release.yml call this script so the rewrite cannot
drift. Nothing here is committed.

Usage (from the repo root, or anywhere; paths are relative to this file):

  python3 scripts/set-crate-versions.py --from-tag v0.1.1
  python3 scripts/set-crate-versions.py --from-release-name app-v0.1.1
  python3 scripts/set-crate-versions.py --version 0.1.1
  python3 scripts/set-crate-versions.py --print-current

If GITHUB_ENV is set (GitHub Actions), VERSION=<semver> is appended so
later steps can read it.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
# Recovery GitHub Release names look like app-v0.1.1: the version is the
# trailing vX.Y.Z. The rest of the name is only there so GitHub does not
# create another v* tag.
RELEASE_NAME_VERSION_RE = re.compile(r"v([0-9]+\.[0-9]+\.[0-9]+)$")
WORKSPACE_CRATES = {"syllabix", "syllabix-core", "syllabix-native"}


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def write_github_env(version: str) -> None:
    env_file = os.environ.get("GITHUB_ENV")
    if not env_file:
        return
    with open(env_file, "a", encoding="utf-8") as fh:
        fh.write(f"VERSION={version}\n")


def version_from_tag(tag: str) -> str:
    if not tag.startswith("v"):
        sys.exit(f"tag {tag!r} must start with v")
    version = tag[1:]
    if not VERSION_RE.fullmatch(version):
        sys.exit(f"tag {tag!r} is not a plain vX.Y.Z version")
    return version


def version_from_release_name(name: str) -> str:
    name = name.strip()
    match = RELEASE_NAME_VERSION_RE.search(name)
    if not match:
        sys.exit(
            f"release_name {name!r} does not end with a vX.Y.Z version"
        )
    return match.group(1)


def current_workspace_version(root: pathlib.Path) -> str:
    in_pkg = False
    for line in (root / "Cargo.toml").read_text().splitlines():
        if line.strip() == "[workspace.package]":
            in_pkg = True
            continue
        if in_pkg and line.startswith("["):
            break
        if in_pkg:
            match = re.match(r'^version\s*=\s*"([^"]+)"\s*$', line)
            if match:
                return match.group(1)
    sys.exit("could not find [workspace.package] version")


def rewrite(root: pathlib.Path, version: str) -> None:
    cargo_toml = root / "Cargo.toml"
    text = cargo_toml.read_text()
    lines = text.splitlines(keepends=True)
    in_pkg = False
    replaced_pkg = False
    out = []
    for line in lines:
        if line.strip() == "[workspace.package]":
            in_pkg = True
            out.append(line)
            continue
        if in_pkg and line.startswith("["):
            in_pkg = False
        if in_pkg and re.match(r"^version\s*=", line) and not replaced_pkg:
            out.append(f'version = "{version}"\n')
            replaced_pkg = True
            continue
        out.append(line)
    if not replaced_pkg:
        sys.exit("could not find [workspace.package] version")
    text = "".join(out)
    text, n = re.subn(
        r'syllabix-native = \{ path = "crates/syllabix-native", version = "[^"]+" \}',
        f'syllabix-native = {{ path = "crates/syllabix-native", version = "{version}" }}',
        text,
        count=1,
    )
    if n != 1:
        sys.exit("could not find syllabix-native version requirement")
    cargo_toml.write_text(text)

    # Keep Cargo.lock in sync so `cargo build --locked` / `cargo publish
    # --locked` still work after the Cargo.toml edit. This is not committed.
    lock_path = root / "Cargo.lock"
    lock_out = []
    pending = None
    lock_n = 0
    for line in lock_path.read_text().splitlines(keepends=True):
        m = re.match(r'^name = "([^"]+)"\s*$', line)
        if m and m.group(1) in WORKSPACE_CRATES:
            pending = m.group(1)
            lock_out.append(line)
            continue
        if pending and re.match(r"^version = ", line):
            lock_out.append(f'version = "{version}"\n')
            lock_n += 1
            pending = None
            continue
        pending = None
        lock_out.append(line)
    if lock_n != 3:
        sys.exit(f"expected 3 workspace crate versions in Cargo.lock, got {lock_n}")
    lock_path.write_text("".join(lock_out))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set workspace crate versions for a release job (not committed)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--from-tag",
        metavar="TAG",
        help="Git tag vX.Y.Z (crates.io publish and tag-push app builds)",
    )
    group.add_argument(
        "--from-release-name",
        metavar="NAME",
        help="Recovery GitHub Release name ending in vX.Y.Z, e.g. app-v0.1.1",
    )
    group.add_argument(
        "--version",
        metavar="X.Y.Z",
        help="Literal X.Y.Z (local rehearsal)",
    )
    group.add_argument(
        "--print-current",
        action="store_true",
        help="Print [workspace.package] version; do not rewrite files",
    )
    args = parser.parse_args()
    root = repo_root()

    if args.print_current:
        version = current_workspace_version(root)
        write_github_env(version)
        print(version)
        return

    if args.from_tag is not None:
        tag = args.from_tag
        version = version_from_tag(tag)
        rewrite(root, version)
        write_github_env(version)
        print(f"tag={tag} publish_version={version}")
        return

    if args.from_release_name is not None:
        name = args.from_release_name
        version = version_from_release_name(name)
        rewrite(root, version)
        write_github_env(version)
        print(f"release_name={name} publish_version={version}")
        return

    version = args.version
    if not VERSION_RE.fullmatch(version):
        sys.exit(f"version {version!r} is not a plain X.Y.Z version")
    rewrite(root, version)
    write_github_env(version)
    print(f"publish_version={version}")


if __name__ == "__main__":
    main()
