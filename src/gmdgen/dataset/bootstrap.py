from pathlib import Path

def initialize_dataset_structure(dataset_dir: str | Path) -> dict[str, list[str]]:
    base = Path(dataset_dir)
    dirs = [
        base / "docs",
        base / "reference_levels" / "modern",
        base / "reference_levels" / "nine_circles",
        base / "reference_levels" / "layout",
        base / "reference_levels" / "effect_heavy",
        base / "reference_levels" / "wave",
        base / "reference_levels" / "demon",
        base / "styles",
        base / "learning" / "examples",
        base / "learning" / "feedback",
        base / "learning" / "exports",
        base / "eval" / "audio",
        base / "eval" / "levels",
        base / "eval" / "reports",
    ]
    created = []
    existed = []
    for d in dirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(str(d))
        else:
            existed.append(str(d))
    
    return {"created": created, "existed": existed}
