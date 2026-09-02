#!/usr/bin/env bash
# Fetch the document corpus: the English FastAPI documentation (MIT licensed).
#
# The corpus is not vendored into this repository -- it is ~1.5 MB of someone else's
# documentation, and committing it would make this repo a fork of theirs. It is fetched
# reproducibly instead, pinned to a tag so a re-run produces the same corpus and evaluation
# numbers stay comparable across machines and across time.
#
# Three paths are checked out, and all three are needed:
#   docs/en/docs  the documentation itself
#   docs_src      the code examples that {* ... *} include directives pull in
#   fastapi       one directive reaches into the library source itself
# Without the latter two, 304 sections lose their code examples and the corpus becomes
# prose -- which would test the hybrid-search thesis with its own evidence removed.
set -euo pipefail

REPO_URL="https://github.com/fastapi/fastapi.git"
REF="${CORPUS_REF:-0.115.6}"
DEST="${1:-data/raw/fastapi}"

if [ -d "$DEST/.git" ]; then
  echo "corpus already present at $DEST (delete it to re-fetch)"
  exit 0
fi

echo "fetching FastAPI docs @ $REF -> $DEST"
mkdir -p "$(dirname "$DEST")"

# --filter=blob:none defers file contents until checkout, and --sparse plus a narrow
# checkout means only the three paths above are ever materialised. A full clone of this
# repository is ~90 MB; this is a small fraction of that.
git clone --quiet --depth 1 --branch "$REF" --filter=blob:none --sparse "$REPO_URL" "$DEST"
git -C "$DEST" sparse-checkout set docs/en/docs docs_src fastapi

echo "fetched $(find "$DEST/docs/en/docs" -name '*.md' | wc -l | tr -d ' ') markdown files"
