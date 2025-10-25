from __future__ import annotations
from pathlib import Path
import re

_ORD_RE = re.compile(r"^(\d+)(st|nd|rd|th)$")

def _ordinal(n: int) -> str:
    if 10 <= (n % 100) <= 13: suf = "th"
    else: suf = {1:"st",2:"nd",3:"rd"}.get(n%10,"th")
    return f"{n}{suf}"

def next_ordinal_run_name(model_dir: str | Path) -> str:
    model_dir = Path(model_dir)
    if not model_dir.exists(): return "1st"
    highest = 0
    for p in model_dir.iterdir():
        if p.is_dir():
            m = _ORD_RE.match(p.name)
            if m: highest = max(highest, int(m.group(1)))
            else: highest = max(highest, 1)
    return _ordinal(highest + 1)
