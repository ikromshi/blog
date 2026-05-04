#!/usr/bin/env python3
"""
Convert Jupyter notebooks to Jekyll collection documents.

Usage:
    python _scripts/convert_notebooks.py

Input:  notebooks/{project}/*.ipynb
Output: _notebooks/{project}/{slug}.md

For each project found under notebooks/, if no projects/{project}.md
page exists yet, one is auto-generated from _data/projects.yml.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_SRC = ROOT / "notebooks"
NOTEBOOKS_OUT = ROOT / "_notebooks"
PROJECTS_DIR = ROOT / "projects"
DATA_FILE = ROOT / "_data" / "projects.yml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-")


def extract_title(cells: list) -> str | None:
    """Return first H2 heading found in any markdown cell."""
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        m = re.search(r"^##\s+(.+)$", source, re.MULTILINE)
        if m:
            return m.group(1).strip()
    return None


def render_output(output: dict) -> str:
    """Render one cell output block to Markdown."""
    parts = []
    otype = output.get("output_type", "")

    if otype == "stream":
        text = "".join(output.get("text", []))
        if text.strip():
            parts.append(f"\n```\n{text.rstrip()}\n```\n")

    elif otype in ("display_data", "execute_result"):
        data = output.get("data", {})
        if "image/png" in data:
            b64 = data["image/png"].replace("\n", "")
            parts.append(f"\n![output](data:image/png;base64,{b64})\n")
        elif "image/svg+xml" in data:
            svg = "".join(data["image/svg+xml"])
            parts.append(f"\n{svg}\n")
        elif "text/html" in data:
            html = "".join(data["text/html"])
            parts.append(f"\n{html}\n")
        elif "text/plain" in data:
            text = "".join(data["text/plain"])
            if text.strip():
                parts.append(f"\n```\n{text.rstrip()}\n```\n")

    elif otype == "error":
        ename = output.get("ename", "Error")
        evalue = output.get("evalue", "")
        tb = output.get("traceback", [])
        # Strip ANSI color codes from traceback
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        clean_tb = [ansi_escape.sub("", line) for line in tb]
        tb_text = "\n".join(clean_tb) if clean_tb else f"{ename}: {evalue}"
        parts.append(f"\n```\n{tb_text}\n```\n")

    return "".join(parts)


def convert_notebook(nb_path: Path, order: int, project_id: str) -> tuple[str, str, str]:
    """
    Convert a single notebook file.
    Returns (slug, title, markdown_content).
    """
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    title = extract_title(cells) or nb_path.stem
    slug = slugify(nb_path.stem)

    # Escape any double-quotes in the title for YAML
    safe_title = title.replace('"', '\\"')

    front_matter = (
        "---\n"
        f'layout: notebook\n'
        f'title: "{safe_title}"\n'
        f'project: {project_id}\n'
        f'order: {order}\n'
        "---\n\n"
    )

    body_parts = []
    for cell in cells:
        ctype = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))

        if ctype == "markdown":
            body_parts.append(source.rstrip() + "\n\n")

        elif ctype == "code":
            if source.strip():
                body_parts.append(f"```python\n{source}\n```\n")
            for output in cell.get("outputs", []):
                rendered = render_output(output)
                if rendered:
                    body_parts.append(rendered)
            body_parts.append("\n")

    return slug, title, front_matter + "".join(body_parts)


def load_project_meta(project_id: str) -> dict:
    """Read _data/projects.yml and return the entry for project_id, or empty dict."""
    if not DATA_FILE.exists():
        return {}
    try:
        # Minimal YAML parser for our simple list-of-dicts format
        import re as _re
        text = DATA_FILE.read_text(encoding="utf-8")
        # Find the block starting with "- id: {project_id}"
        pattern = _re.compile(
            rf"-\s+id:\s+{_re.escape(project_id)}\n((?:\s+\S[^\n]*\n)*)",
            _re.MULTILINE,
        )
        m = pattern.search(text)
        if not m:
            return {}
        block = m.group(0)
        result = {"id": project_id}
        for key in ("title", "description"):
            km = _re.search(rf'^\s+{key}:\s+"?(.+?)"?\s*$', block, _re.MULTILINE)
            if km:
                result[key] = km.group(1).strip().strip('"')
        return result
    except Exception:
        return {}


def ensure_project_page(project_id: str, meta: dict) -> None:
    """Create projects/{project_id}.md if it doesn't already exist."""
    page_path = PROJECTS_DIR / f"{project_id}.md"
    if page_path.exists():
        return
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    title = meta.get("title", project_id.replace("-", " ").title())
    desc = meta.get("description", "")
    safe_title = title.replace('"', '\\"')
    safe_desc = desc.replace('"', '\\"')
    content = (
        "---\n"
        "layout: project\n"
        f'title: "{safe_title}"\n'
        f"project: {project_id}\n"
        f'description: "{safe_desc}"\n'
        f"permalink: /projects/{project_id}/\n"
        "---\n"
    )
    page_path.write_text(content, encoding="utf-8")
    print(f"  Created projects/{project_id}.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not NOTEBOOKS_SRC.exists():
        print("notebooks/ directory not found — nothing to convert.")
        return

    project_dirs = sorted(d for d in NOTEBOOKS_SRC.iterdir() if d.is_dir())
    if not project_dirs:
        print("No project directories found under notebooks/.")
        return

    for project_dir in project_dirs:
        project_id = project_dir.name
        notebooks = sorted(project_dir.glob("*.ipynb"))
        if not notebooks:
            continue

        print(f"\nProject '{project_id}': {len(notebooks)} notebook(s)")

        meta = load_project_meta(project_id)
        ensure_project_page(project_id, meta)

        out_dir = NOTEBOOKS_OUT / project_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Remove stale generated files for notebooks that no longer exist
        existing_slugs = set()
        for order, nb_path in enumerate(notebooks, start=1):
            existing_slugs.add(slugify(nb_path.stem) + ".md")
        for stale in out_dir.glob("*.md"):
            if stale.name not in existing_slugs:
                stale.unlink()
                print(f"  Removed stale: {stale.name}")

        for order, nb_path in enumerate(notebooks, start=1):
            try:
                slug, title, content = convert_notebook(nb_path, order, project_id)
                out_path = out_dir / f"{slug}.md"
                out_path.write_text(content, encoding="utf-8")
                print(f"  [{order:02d}] {title[:60]}")
            except Exception as exc:
                print(f"  [ERROR] {nb_path.name}: {exc}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
