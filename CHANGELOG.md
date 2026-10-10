# Changelog

The CLI (`syllabix`) and the SDK (`syllabix-core`) share one version. One
`[workspace.package] version` in the root `Cargo.toml` and one git tag
`vX.Y.Z` cover both, plus the GitHub Release binaries. Other repos pin
`tag = "vX.Y.Z"` on the git dependency and do not track `main`. How to
depend: [docs/embed.md](docs/embed.md).

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Dates are UTC.

## Unreleased

### Added

- Tag and changelog contract: SDK and CLI version together; dependents pin
  a `vX.Y.Z` git tag.
- Host compile and model-cache sharing for repos that embed `syllabix-core`
  (`docs/embed.md`, `docs/install.md`).

## 0.1.0 - 2026-09-29

First tag other repos should pin (`workspace.package.version` is `0.1.0`).
Lane 1 binaries are the GitHub Release artifacts for this tag.
