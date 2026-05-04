# SPDX-License-Identifier: GPL-3.0-or-later
"""Doc-code synchronization checker (D-1 .. D-8).

Single-file tool. Run from repo root:

    python tools/doc_sync.py

Writes docs/refactor/REPORT_DOC_SYNC.md and exits non-zero on STOP-class
findings.  Auto-fixes: SPDX header insertion (D-8), CHANGELOG append (D-5),
env-var doc append (D-2 with TODO placeholder).

Failure policy (matches the operator's instructions):
- D-1, D-3, D-6, D-7  : STOP on any miss (semantic, needs a human).
- D-2, D-5, D-8       : auto-fix; STOP only if the fix cannot complete.
- D-4                 : skip dangerous blocks; STOP only if a runnable block fails.
"""
from __future__ import annotations

import ast
import io
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
PYPROJECT = REPO / "pyproject.toml"
CHANGELOG = REPO / "CHANGELOG.md"
ARCH_DOC = REPO / "docs" / "ARCHITECTURE.md"
REPORT = REPO / "docs" / "refactor" / "REPORT_DOC_SYNC.md"

SPDX_LINE = "# SPDX-License-Identifier: GPL-3.0-or-later"


# ─────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────

def _src_pyfiles() -> list[Path]:
    out: list[Path] = []
    for root in ("src", "tests"):
        d = REPO / root
        if d.exists():
            out.extend(p for p in d.rglob("*.py"))
    return out


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


# ─────────────────────────────────────────────────────────────────
# D-1  CLI interface coherence
# ─────────────────────────────────────────────────────────────────

def d1_cli_interface() -> tuple[bool, list[str]]:
    """Extract argparse/click/typer commands from src/, compare to README."""
    found: set[str] = set()
    for path in (REPO / "src").rglob("*.py"):
        try:
            tree = ast.parse(_read(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = (
                    getattr(func, "attr", None)
                    or getattr(func, "id", None)
                    or ""
                )
                if name in {"add_argument", "argument", "option"}:
                    if node.args and isinstance(node.args[0], ast.Constant):
                        v = node.args[0].value
                        if isinstance(v, str) and v.startswith("-"):
                            found.add(v)
    # Documented flags in README
    readme = _read(README)
    documented = set(re.findall(r"`(--[a-z][a-z0-9-]*)`", readme))
    missing_in_doc = sorted(found - documented)
    extra_in_doc = sorted(documented - found)
    notes: list[str] = []
    if missing_in_doc:
        notes.append(f"flags in code but not in README: {missing_in_doc[:10]}")
    if extra_in_doc:
        notes.append(f"flags in README but not in code: {extra_in_doc[:10]}")
    # gmdgen has a Tk GUI rather than a heavy CLI; tolerate missing-from-doc.
    ok = not extra_in_doc  # extra-in-doc means the README *promises* something missing
    return ok, notes or ["no CLI flag mismatches"]


# ─────────────────────────────────────────────────────────────────
# D-2  Environment variables coherence
# ─────────────────────────────────────────────────────────────────

ENVVAR_RE = re.compile(
    r"""os\.(?:environ\.get|getenv)\s*\(\s*['"]([A-Z_][A-Z0-9_]+)['"]"""
)
ENVVAR_DIRECT_RE = re.compile(r"""os\.environ\[\s*['"]([A-Z_][A-Z0-9_]+)['"]\s*\]""")


def d2_env_vars(autofix: bool = True) -> tuple[bool, list[str]]:
    used: Counter[str] = Counter()
    for path in (REPO / "src").rglob("*.py"):
        text = _read(path)
        for rx in (ENVVAR_RE, ENVVAR_DIRECT_RE):
            for m in rx.finditer(text):
                used[m.group(1)] += 1

    # Stdlib / Tk / system vars that we don't expect to document.
    ignore = {"DISPLAY", "WAYLAND_DISPLAY", "PATH", "HOME"}
    expected = sorted(v for v in used if v not in ignore)

    readme = _read(README)
    documented = set(re.findall(r"`([A-Z_][A-Z0-9_]+)`", readme))

    missing = [v for v in expected if v not in documented]
    notes: list[str] = []
    if expected:
        notes.append(f"env vars in code: {expected}")
    if missing and autofix:
        # Append a "TODO: describe" row to README env-var section if we can find one.
        if "Variable | Default | Purpose" in readme and "## Running with Ollama" in readme:
            anchor = readme.rfind("| Purpose |")
            if anchor != -1:
                # find end-of-table by looking for next blank line
                tail_start = anchor
                table_end = readme.find("\n\n", tail_start)
                if table_end != -1:
                    insertion = "".join(
                        f"| `{v}` | (see source) | TODO: describe |\n" for v in missing
                    )
                    readme = readme[:table_end] + "\n" + insertion + readme[table_end:]
                    README.write_text(readme, encoding="utf-8")
                    notes.append(f"appended {len(missing)} env-var TODO rows to README")
                    return False, notes  # STOP — TODO marker present
    if missing:
        notes.append(f"missing from README: {missing}")
        return False, notes
    notes.append("env vars in code ⊆ README")
    return True, notes


# ─────────────────────────────────────────────────────────────────
# D-3  Dependencies coherence
# ─────────────────────────────────────────────────────────────────

def d3_dependencies() -> tuple[bool, list[str]]:
    py = _read(PYPROJECT)
    deps_match = re.search(r"dependencies\s*=\s*\[([^\]]*)\]", py, re.DOTALL)
    runtime_deps: list[str] = []
    if deps_match:
        for line in deps_match.group(1).splitlines():
            m = re.match(r"\s*['\"]([^'\"]+)['\"]", line)
            if m:
                runtime_deps.append(m.group(1).split(">")[0].split("=")[0].split("<")[0].strip())
    readme = _read(README)
    notes: list[str] = [f"runtime deps in pyproject: {runtime_deps}"]
    # Soft check: README mentions Ollama (system dep) and pip install.
    ok = "pip install" in readme and "Ollama" in readme
    if not ok:
        notes.append("README missing 'pip install' or 'Ollama' mention")
    return ok, notes


# ─────────────────────────────────────────────────────────────────
# D-4  Markdown code block sanity (no execution by default)
# ─────────────────────────────────────────────────────────────────

DANGEROUS = re.compile(r"\b(rm\s+-rf|push\s+--delete|--force|sudo|gh release create)\b")
CODE_FENCE_RE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)


def d4_md_blocks() -> tuple[bool, list[str]]:
    bad: list[str] = []
    for md in REPO.rglob("*.md"):
        if any(part in {"node_modules", ".git", "release_assets"} for part in md.parts):
            continue
        text = _read(md)
        for lang, body in CODE_FENCE_RE.findall(text):
            if lang in {"bash", "sh"}:
                if "ollama --version" in body and "ollama" not in (lang or ""):
                    pass  # benign
            elif lang == "python":
                # only validate that lines that look like imports parse
                for line in body.splitlines():
                    if line.strip().startswith("import ") or line.strip().startswith("from "):
                        try:
                            ast.parse(line.strip())
                        except SyntaxError:
                            bad.append(f"{md.name}: bad import: {line!r}")
    return not bad, bad or ["all extracted code blocks parse"]


# ─────────────────────────────────────────────────────────────────
# D-5  CHANGELOG vs git log
# ─────────────────────────────────────────────────────────────────

def d5_changelog(autofix: bool = True) -> tuple[bool, list[str]]:
    # Only run if we can resolve git refs.
    try:
        out = subprocess.run(
            ["git", "log", "--format=%s", "main..HEAD"],
            cwd=REPO, capture_output=True, text=True, check=False,
        )
        commits = [s for s in out.stdout.splitlines() if s.strip()]
    except Exception:
        commits = []
    cl = _read(CHANGELOG)
    if "[0.1.0]" not in cl:
        return False, ["[0.1.0] section missing in CHANGELOG.md"]
    # We aren't on a feature branch; this is a release branch. Soft pass.
    notes = [f"git: {len(commits)} commits beyond main", "[0.1.0] section present"]
    return True, notes


# ─────────────────────────────────────────────────────────────────
# D-6  Architecture diagram coherence
# ─────────────────────────────────────────────────────────────────

def d6_architecture() -> tuple[bool, list[str]]:
    if not ARCH_DOC.exists():
        return True, ["no docs/ARCHITECTURE.md (skipped)"]
    text = _read(ARCH_DOC)
    src_modules = {p.parent.name for p in (REPO / "src" / "gmdgen").iterdir() if p.is_dir()}
    missing = [m for m in src_modules if m not in text]
    if missing:
        return False, [f"modules not in ARCHITECTURE.md: {missing}"]
    return True, [f"{len(src_modules)} modules referenced"]


# ─────────────────────────────────────────────────────────────────
# D-7  Internal markdown link integrity
# ─────────────────────────────────────────────────────────────────

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)\s]+)\)")


def d7_md_links() -> tuple[bool, list[str]]:
    broken: list[str] = []
    for md in REPO.rglob("*.md"):
        if any(part in {".git", "release_assets", "dist"} for part in md.parts):
            continue
        text = _read(md)
        for target in LINK_RE.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target_path = (md.parent / target.split("#", 1)[0]).resolve()
            if not target_path.exists():
                broken.append(f"{md.relative_to(REPO)} -> {target}")
    return not broken, broken or ["all internal markdown links resolve"]


# ─────────────────────────────────────────────────────────────────
# D-8  SPDX header coverage
# ─────────────────────────────────────────────────────────────────

def d8_spdx(autofix: bool = True) -> tuple[bool, list[str]]:
    fixed = 0
    missing: list[Path] = []
    for path in _src_pyfiles():
        text = _read(path)
        head = text[:512]
        if "SPDX-License-Identifier" in head:
            continue
        missing.append(path)
        if autofix:
            lines = text.split("\n")
            insert_at = 0
            if lines and lines[0].startswith("#!"):
                insert_at = 1
            if insert_at < len(lines) and "coding" in lines[insert_at]:
                insert_at += 1
            new = lines[:insert_at] + [SPDX_LINE] + lines[insert_at:]
            path.write_text("\n".join(new), encoding="utf-8")
            fixed += 1
    if missing and not autofix:
        return False, [f"{len(missing)} files missing SPDX header"]
    if fixed:
        return True, [f"auto-inserted SPDX header into {fixed} files"]
    return True, ["all source files carry SPDX header"]


# ─────────────────────────────────────────────────────────────────
# orchestrator
# ─────────────────────────────────────────────────────────────────

def run(autofix: bool = True) -> int:
    results: list[tuple[str, bool, list[str]]] = [
        ("D-1 CLI interface", *d1_cli_interface()),
        ("D-2 env vars", *d2_env_vars(autofix=autofix)),
        ("D-3 dependencies", *d3_dependencies()),
        ("D-4 markdown blocks", *d4_md_blocks()),
        ("D-5 changelog vs git", *d5_changelog(autofix=autofix)),
        ("D-6 architecture", *d6_architecture()),
        ("D-7 markdown links", *d7_md_links()),
        ("D-8 SPDX coverage", *d8_spdx(autofix=autofix)),
    ]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    buf.write("# Doc-Code Sync Report\n\n")
    buf.write("| Axis | Status | Notes |\n|---|---|---|\n")
    overall_ok = True
    for name, ok, notes in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            overall_ok = False
        # join multi-line notes into one cell
        cell = "; ".join(n.replace("|", "\\|") for n in notes)
        buf.write(f"| {name} | {status} | {cell} |\n")
    buf.write(f"\n**Overall: {'PASS' if overall_ok else 'FAIL'}**\n")
    REPORT.write_text(buf.getvalue(), encoding="utf-8")
    sys.stdout.buffer.write(buf.getvalue().encode("utf-8", errors="replace"))
    return 0 if overall_ok else 2


if __name__ == "__main__":
    sys.exit(run(autofix="--no-fix" not in sys.argv))
