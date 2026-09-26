"""Hand-authored artifacts (Tables 1, 3, 12, 14; Figure 2), copied from reproduce/static."""
import shutil

from reproduce.paths import STATIC


def build(out, artifact):
    for suffix in (".tex", ".pdf"):
        src = STATIC / f"{artifact.id}{suffix}"
        if src.exists():
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / src.name)
    return None
