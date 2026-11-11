#!/bin/sh
# Publish the hosted read-only viewer: the committed code (HEAD) plus a fresh
# snapshot of the master DB, pushed to the PRIVATE hosting repo that Streamlit
# Community Cloud deploys (main file: viewer.py).
#
#     ./publish_viewer.sh
#
# The data never enters this (public) repository: the snapshot is written to
# viewer_data/ (gitignored here) and committed only in a separate checkout of
# the hosting repo, kept beside this one. Run it after approving new papers.
#
#     VIEWER_REPO      the hosting repo  (default: NicholasHat/ree-viewer)
#     VIEWER_CHECKOUT  its local clone   (default: ../ree-viewer)
set -eu

cd "$(dirname "$0")"
repo=${VIEWER_REPO:-NicholasHat/ree-viewer}
checkout=${VIEWER_CHECKOUT:-../ree-viewer}
python=${PYTHON:-.venv/bin/python}

visibility=$(gh repo view "$repo" --json visibility --jq .visibility)
if [ "$visibility" != "PRIVATE" ]; then
    echo "refusing to publish: $repo is $visibility, not PRIVATE" >&2
    exit 1
fi

summary=$("$python" -m database.snapshot)
echo "$summary"

[ -d "$checkout/.git" ] || gh repo clone "$repo" "$checkout"

# Replace the hosting checkout's contents with exactly HEAD + the snapshot.
git -C "$checkout" rm -rqf --ignore-unmatch .
git archive HEAD | tar -x -C "$checkout"
mkdir -p "$checkout/viewer_data"
cp viewer_data/master.db "$checkout/viewer_data/master.db"
git -C "$checkout" add -A
git -C "$checkout" add -f viewer_data/master.db

if git -C "$checkout" diff --cached --quiet; then
    echo "Nothing new to publish."
    exit 0
fi
git -C "$checkout" commit -q -m "publish: ${summary#*: } (code $(git rev-parse --short HEAD))"
git -C "$checkout" push -q origin HEAD:main
echo "Published to $repo. Streamlit Cloud redeploys within a minute or two."
