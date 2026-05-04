from pathlib import Path
import re

path = Path("src/gmdgen/gui/app.py")

if not path.exists():
    raise SystemExit("src/gmdgen/gui/app.py 파일을 찾을 수 없습니다.")

text = path.read_text(encoding="utf-8", errors="replace")

if "self.v_use_environment_key" not in text:
    print("[OK] v_use_environment_key 참조가 없습니다. 패치 불필요.")
    raise SystemExit(0)

if re.search(r"self\.v_use_environment_key\s*=", text):
    print("[OK] v_use_environment_key가 이미 정의되어 있습니다. 패치 불필요.")
    raise SystemExit(0)

lines = text.splitlines()
inserted = False

for i, line in enumerate(lines):
    if "self.root" in line and "=" in line:
        indent = line[: len(line) - len(line.lstrip())]
        lines.insert(i + 1, indent + "# Compatibility flag used by older GUI validation paths.")
        lines.insert(i + 2, indent + "self.v_use_environment_key = tk.BooleanVar(value=True)")
        inserted = True
        break

if not inserted:
    for i, line in enumerate(lines):
        if re.match(r"^\s*def __init__\s*\(", line):
            indent = " " * (len(line) - len(line.lstrip()) + 4)
            lines.insert(i + 1, indent + "# Compatibility flag used by older GUI validation paths.")
            lines.insert(i + 2, indent + "self.v_use_environment_key = tk.BooleanVar(value=True)")
            inserted = True
            break

if not inserted:
    raise SystemExit("패치 위치를 찾지 못했습니다. app.py의 __init__ 구조를 확인해야 합니다.")

path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("[OK] patched src/gmdgen/gui/app.py")
