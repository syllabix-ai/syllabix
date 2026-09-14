---
name: repository-guide
description: Follow the repository's development and verification conventions.
---

# Repository orientation

When the user asks to be caught up on this repository, use the generic shell
in two foreground steps:

1. Read enough of `README.md` to identify what the repository is for.
   For example: `sed -n '1,160p' README.md`.
2. Inspect a bounded recent slice of Git history using the repository's normal
   history command. Choose a useful bounded range; do not assume a fixed
   commit count. For example: `git log --oneline --decorate --no-merges
   --max-count=<useful-bound>`.

Then answer in two or three spoken sentences: explain the repository's purpose
and mention at least one recent change. Do not include URLs, raw diffs, shell
commands, tool traces, or policy details in the spoken answer.

Keep the default user path tool-free, validate the smallest relevant test set,
and report any verification that could not be run.
