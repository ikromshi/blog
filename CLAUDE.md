# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
bundle install

# Local development server (hot-reload, no GA)
bundle exec jekyll serve

# Build for production
JEKYLL_ENV=production bundle exec jekyll build

# Build and serve with production env (enables GA)
JEKYLL_ENV=production bundle exec jekyll serve
```

## Architecture

This is a Jekyll 4 static blog using the **minima** theme, deployed to GitHub Pages via `.github/workflows/jekyll-gh-pages.yml` on every push to `main`.

**Theme overrides** live alongside the defaults:
- `_layouts/home.html` — overrides the minima home layout to show post excerpts
- `_includes/head.html` — overrides minima head to add favicon and conditionally load GA
- `_includes/google-analytics.html` — GA4 snippet (only injected in `production` env)
- `assets/main.scss` — imports minima then applies custom blue header/footer (`#2f6299`)

**Content:**
- `_posts/` — blog posts in Markdown with front matter (`layout`, `title`, `date`, `excerpt`)
- Posts use kramdown with LaTeX math support (rendered via `$$...$$` syntax)
- `about.markdown` and `index.markdown` are the two static pages

**Configuration (`_config.yml`):**
- GA measurement ID: `G-XLSXNWYKYP` (stored in `site.google_analytics`)
- GA is only active when `JEKYLL_ENV=production`; local `serve` skips it by default
- Plugins: `jekyll-feed` (RSS at `/feed.xml`) and `jekyll-sitemap`

**Deployment:** Pushing to `main` triggers the GitHub Actions workflow which builds with the `actions/jekyll-build-pages` action and deploys to GitHub Pages.
