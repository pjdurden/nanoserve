# nanoserve docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — what gets built and how the pieces fit, with diagrams.
- [PLAN.md](PLAN.md) — the 100-day weekly plan.
- [daily/](daily/) — one short doc per day: what was added, why, what was learned. Start from [daily/TEMPLATE.md](daily/TEMPLATE.md).
- [diagrams/](diagrams/) — SVG diagrams, sized 1200x675 (16:9) for LinkedIn.

## Using the diagrams on LinkedIn

LinkedIn posts need a raster image (PNG or JPG), not an SVG. Two ways to get one:

1. **Screenshot.** Open the `.svg` in a browser, zoom to fit, screenshot it. Fastest.
2. **Convert to PNG.** Pick whichever tool you have:

```bash
# rsvg-convert (from librsvg)
rsvg-convert -w 2400 docs/diagrams/paged-kv-cache.svg -o paged-kv-cache.png

# or cairosvg (pip install cairosvg)
cairosvg docs/diagrams/paged-kv-cache.svg -o paged-kv-cache.png --output-width 2400

# or headless Chrome / Inkscape if you prefer
inkscape docs/diagrams/paged-kv-cache.svg --export-type=png -w 2400
```

`-w 2400` exports at 2x for a crisp feed image. The diagrams use a white background so they sit cleanly in the LinkedIn feed.

## Style of a daily doc

Keep it short. The point is a postable artifact every day, not an essay. A diff, a graph, one thing learned. Each daily doc maps to one post.
