# femto website

Hugo site, deployed to GitHub Pages by `.github/workflows/pages.yml`.

```
content/             markdown pages (examples/<name>.md: front matter `app:` names the viewer)
layouts/             two-column layout, example cards, {{< example >}} shortcode
static/img/examples/ thumbnails, <name>.png (a placeholder is shown until one exists)
```

The interactive viewers are not checked in: CI builds the wasm module and copies
`build-web-mt/examples/html/` to `static/app/` and `examples/html/enable-threads.js`
to `static/`. To preview locally with the viewers:

```sh
cmake --build build-web-mt --target femto_examples_html
mkdir -p site/static/app && cp -r build-web-mt/examples/html/. site/static/app/
cp examples/html/enable-threads.js site/static/
cd site && hugo server      # without the copy step the pages work, the iframes 404
```
