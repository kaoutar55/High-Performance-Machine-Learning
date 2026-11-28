"""Generate the chapter pages of the documentation site from the chapter READMEs.

  python tools/docs/build_pages.py && mkdocs serve

Writes docs/book/<chapter>.md (one page per chapter folder, from its README.md) and copies each chapter's
notebooks to docs/notebooks/<chapter>/, where mkdocs-jupyter renders them. Relative links in the READMEs are
rewritten: chapter READMEs and notebooks link to their site pages, everything else to the file on GitHub.
The generated folders are not committed (see docs/.gitignore).
"""
import json
import os
import re
import shutil
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOCS = os.path.join(ROOT, "docs")
REPO = "kaoutar55/High-Performance-Machine-Learning"
BRANCH = "main"
GH = f"https://github.com/{REPO}"
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}"

# Folders with a page on the site, in book order. Chapter 11's code is not in the repository yet.
PAGES = [f"chapter{i:02d}" for i in range(1, 13)] + ["case_study"]
TITLES_IF_MISSING = {"chapter11": "Chapter 11: Navigating the Accelerator and Compiler Landscape"}
LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)(\s+\"[^\"]*\")?\)")


def tracked_files():
    """Paths tracked by git, or None outside a git checkout. Untracked working notebooks stay off the site."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return set(out.splitlines())


TRACKED = tracked_files()


def notebooks(folder):
    d = os.path.join(ROOT, folder)
    if not os.path.isdir(d):
        return []
    names = sorted(f for f in os.listdir(d) if f.endswith(".ipynb"))
    return [f for f in names if TRACKED is None or f"{folder}/{f}" in TRACKED]


def site_target(folder, target):
    """Rewrite one relative link of folder/README.md; return the new target."""
    if re.match(r"^[a-z]+:|^#|^/", target):
        return target
    path, _, frag = target.partition("#")
    abs_path = os.path.normpath(os.path.join(ROOT, folder, path))
    rel = os.path.relpath(abs_path, ROOT).replace(os.sep, "/")
    parts = rel.split("/")
    if parts[0] in PAGES and (len(parts) == 1 or parts[1:] == ["README.md"] or parts[1:] == [""]):
        return f"{parts[0]}.md" + (f"#{frag}" if frag else "")
    if len(parts) == 2 and parts[0] in PAGES and parts[1].endswith(".ipynb"):
        return f"../notebooks/{parts[0]}/{parts[1]}"
    kind = "tree" if os.path.isdir(abs_path) else "blob"
    if not os.path.exists(abs_path):
        kind = "blob"
    return f"{GH}/{kind}/{BRANCH}/{rel}" + (f"#{frag}" if frag else "")


def rewrite(folder, text):
    def sub(m):
        bang, label, target, title = m.group(1), m.group(2), m.group(3), m.group(4) or ""
        new = site_target(folder, target)
        if bang and new.startswith(GH + "/blob/"):          # images: raw file
            new = new.replace(f"{GH}/blob/", f"https://raw.githubusercontent.com/{REPO}/")
        return f"{bang}[{label}]({new}{title})"
    return LINK.sub(sub, text)


def page(folder):
    readme = os.path.join(ROOT, folder, "README.md")
    if os.path.exists(readme):
        text = open(readme, encoding="utf-8").read()
        lines = text.splitlines()
        title, body = lines[0], "\n".join(lines[1:])
    else:
        title, body = "# " + TITLES_IF_MISSING.get(folder, folder), (
            "\nThe companion code for this chapter will be published in this repository. "
            "The [experiment map](../experiment-map.md) summarizes what the chapter measures.\n")
    header = [title, "", f"[:fontawesome-brands-github: Code for this chapter on GitHub]({GH}/tree/{BRANCH}/{folder})"
              "{ .md-button }", ""]
    nbs = notebooks(folder)
    if nbs:
        header += ["## Notebooks", "", "| Notebook | Read here | Run |", "|---|---|---|"]
        for nb in nbs:
            header.append(f"| `{nb}` | [Rendered notebook](../notebooks/{folder}/{nb}) | "
                          f"[Open in Colab]({COLAB}/{folder}/{nb}) · [GitHub]({GH}/blob/{BRANCH}/{folder}/{nb}) |")
        header.append("")
    return "\n".join(header) + "\n" + rewrite(folder, body) + "\n"


def main():
    for sub in ("book", "notebooks"):
        shutil.rmtree(os.path.join(DOCS, sub), ignore_errors=True)
        os.makedirs(os.path.join(DOCS, sub))
    for folder in PAGES:
        open(os.path.join(DOCS, "book", f"{folder}.md"), "w", encoding="utf-8").write(page(folder))
        for nb in notebooks(folder):
            os.makedirs(os.path.join(DOCS, "notebooks", folder), exist_ok=True)
            src = os.path.join(ROOT, folder, nb)
            json.load(open(src, encoding="utf-8"))               # fail early on a broken notebook
            shutil.copy(src, os.path.join(DOCS, "notebooks", folder, nb))
    print("pages:", len(PAGES), "notebooks:", sum(len(notebooks(f)) for f in PAGES))


if __name__ == "__main__":
    main()
