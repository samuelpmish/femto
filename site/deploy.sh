#!/usr/bin/env bash
# Build the site (with the wasm viewers from build-web-mt) and publish it to the
# gh-pages branch as a single fresh commit, so history never accumulates the
# ~12 MB of wasm and meshes.  Pages setting: Source = "Deploy from a branch",
# gh-pages, / (root).  Run from anywhere:   site/deploy.sh   (DRY=1 to skip the push)
set -euo pipefail
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
web=$root/build-web-mt/examples/html
domain=femto.smish.dev

[ -f "$web/femto_examples.wasm" ] || { echo "build first: cmake --build build-web-mt --target femto_examples_html" >&2; exit 1; }

# stage the viewers, as site/README.md describes
mkdir -p "$root/site/static/app"
cp -r "$web"/. "$root/site/static/app/"
rm -rf "$root"/site/static/app/{smoke_examples.js,serve.py,CMakeFiles,cmake_install.cmake}
cp "$root/examples/html/enable-threads.js" "$root/site/static/enable-threads.js"

out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
(cd "$root/site" && hugo --minify --destination "$out")
echo "$domain" > "$out/CNAME"   # a branch deploy needs the domain in the published root
touch "$out/.nojekyll"

if [ -n "${DRY:-}" ]; then echo "dry run: built $(du -sh "$out" | cut -f1) in $out"; cp -r "$out" "$root/site/public-dry"; exit 0; fi

cd "$out"
git init -q -b gh-pages
git add -A
git commit -q -m "Deploy site from $(git -C "$root" rev-parse --short HEAD)"
git push -f "$(git -C "$root" remote get-url origin)" gh-pages
