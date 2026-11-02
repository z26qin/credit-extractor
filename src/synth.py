"""Synthetic credit excerpts with a known answer, for SFT and RLVR (docs/next_steps.md).

An excerpt mixes clause passages and look-alikes from configs/synth_bank.yaml into paragraphs
of CUAD commercial contracts. The answer is the clause passages themselves, so no labeling is
needed and a reward can check correctness, not just copying. Excerpts copy two faults of the
M1 text: curly quote marks and page numbers that break a passage in two.

    python -m src.synth --cuad-json CUADv1.json --out <dir> --text-dir <M1 text dir>
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import yaml

from .cuad import contract_text, load_contracts
from .metrics import match_key
from .parsing import norm_ws
from .prompts import REPO_ROOT, credit_clauses

Example = dict      # {id, excerpt, answer, gold}; gold is {clause: [{text, main}]}

SET_NAMES = ("train", "rlvr", "held_out")
ITEM = "{item} "
SLOT = re.compile(r"\{\w+\}")
CREDIT_WORDS = re.compile(
    r"indebtedness|\bliens?\b|restricted payment|applicable (?:margin|rate)|maturity|incremental"
    r"|most favou?red|leverage ratio|interest coverage|fixed charge|ebitda|available amount"
    r"|unrestricted subsidiar|guarantor", re.I)
MIN_FILLER_CHARS, MAX_FILLER_CHARS = 80, 1500
LEAK_WORDS = 15            # longest run of bank words allowed to match an M1 agreement
NEAR_SHARE = 0.7           # share of look-alikes placed beside a clause passage
PAGE_EVERY_CHARS = 2500

BORROWERS = ("the Borrower", "the Company", "the Parent Borrower", "the Lead Borrower")
HOLDINGS = ("Holdings", "Parent", "Topco")
AGENTS = ("Administrative Agent", "Agent")
EBITDAS = ("Consolidated EBITDA", "Consolidated Adjusted EBITDA", "LTM EBITDA", "Adjusted EBITDA")
RATIOS = ("First Lien Net Leverage Ratio", "Total Net Leverage Ratio",
          "Secured Net Leverage Ratio", "Consolidated Leverage Ratio")
RATES = ("Term SOFR", "Adjusted Term SOFR")
BASES = ("Base Rate", "ABR", "Alternate Base Rate")
LOANS = ("Term Loans", "Initial Term Loans", "Tranche B Term Loans", "Term B Loans")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December")
ORDINALS = ("fifth", "sixth", "seventh")
ITEMS = tuple("abcdefghijklmnopqrstuvwxyz") + ("ii", "iv", "vi", "ix", "xi", "xii", "xiv", "xv")


def load_bank(path: Path | None = None) -> dict:
    path = path or REPO_ROOT / "configs" / "synth_bank.yaml"
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def split_bank(bank: dict) -> tuple[dict, dict]:
    """(training bank, held-out bank). Held-out wording never appears in a training excerpt."""
    def part(held: bool) -> dict:
        tail = lambda xs, n=1: xs[-n:] if held else xs[:-n]
        return {"multi": bank["multi"],
                "clauses": {c: tail(ts) for c, ts in bank["clauses"].items()},
                "lookalikes": {c: tail(ts) for c, ts in bank["lookalikes"].items()},
                "general": tail(bank["general"], 2)}
    return part(False), part(True)


class Slots(dict):
    """Values for one excerpt: names stay fixed, numbers change with each passage."""

    def __init__(self, rng: random.Random):
        super().__init__(
            borrower=rng.choice(BORROWERS), holdings=rng.choice(HOLDINGS),
            agent=rng.choice(AGENTS), ebitda=rng.choice(EBITDAS), ratio_name=rng.choice(RATIOS),
            rate=rng.choice(RATES), base=rng.choice(BASES), loans=rng.choice(LOANS))
        self["Borrower"] = "T" + self["borrower"][1:]
        self["Holdings"] = self["holdings"]
        self.rng = rng

    def new_numbers(self) -> None:
        r = self.rng
        year = r.randint(2027, 2029)
        amounts = sorted(r.sample(range(10, 400, 5), 2), reverse=True)
        ratios = sorted(r.sample(range(250, 625, 25), 3), reverse=True)
        margins = sorted(r.sample(range(150, 475, 25), 3), reverse=True)
        self.update(
            amount=f"${amounts[0] * 1_000_000:,}", amount2=f"${amounts[1] * 1_000_000:,}",
            pct=f"{r.choice(range(10, 105, 5))}.0%", pct2=f"{r.choice((50, 75, 100))}%",
            ratio=f"{ratios[0] / 100:.2f}:1.00", ratio2=f"{ratios[1] / 100:.2f}:1.00",
            ratio3=f"{ratios[2] / 100:.2f}:1.00", cover=f"{r.choice((1.10, 1.25, 2.00, 2.50, 3.00)):.2f}:1.00",
            margin=f"{margins[0] / 100:.2f}%", margin2=f"{margins[0] / 100 - 1:.2f}%",
            margin3=f"{margins[1] / 100:.2f}%", margin4=f"{margins[1] / 100 - 1:.2f}%",
            margin5=f"{margins[2] / 100:.2f}%", margin6=f"{margins[2] / 100 - 1:.2f}%",
            fee=f"{r.choice((0.25, 0.375, 0.50, 1.00, 2.00))}%", bps=r.choice((50, 75, 100)),
            months=r.choice((6, 12, 18, 24)), days=r.choice((91, 120, 180)),
            count=r.choice(("two", "three", "five")),
            year=year, year2=year + 1, nth=r.choice(ORDINALS),
            date=f"{r.choice(MONTHS)} {r.randint(1, 28)}, {r.randint(2030, 2033)}",
            sec=f"{r.randint(2, 10)}.{r.randint(1, 19):02d}",
            sec2=f"{r.randint(2, 10)}.{r.randint(1, 19):02d}", art=r.choice(("III", "IV", "V", "VII")),
            page=r.randint(2, 180), item=f"({r.choice(ITEMS)})")

    def fill(self, template: str) -> tuple[str, str]:
        """(text in the excerpt, text in the answer), which differ by the clause label."""
        self.new_numbers()
        body = template.removeprefix(ITEM)
        text = body.format_map(self)
        return (f"{self['item']} {text}" if template.startswith(ITEM) else text), text


def filler_groups(contracts: dict[str, dict], titles: list[str]) -> list[list[str]]:
    """Paragraphs of each contract in order, without any that sound like a credit agreement."""
    groups = []
    for title in titles:
        kept = [norm_ws(p) for p in re.split(r"\n\s*\n", contract_text(contracts[title]))]
        kept = [p for p in kept if MIN_FILLER_CHARS <= len(p) <= MAX_FILLER_CHARS
                and not CREDIT_WORDS.search(p)]
        if kept:
            groups.append(kept)
    return groups


def break_page(text: str, page: int, rng: random.Random) -> str:
    """Put a page number at a space in the middle of a passage."""
    spaces = [m.start() for m in re.finditer(" ", text)]
    at = rng.choice(spaces[len(spaces) // 4:3 * len(spaces) // 4])
    return f"{text[:at]}\n\n{page}\n\n{text[at + 1:]}"


def pick_passages(pool: dict, clauses: list[str], slots: Slots, rng: random.Random,
                  spec: dict) -> list[tuple[str, str, str]]:
    """(clause, excerpt text, answer text) for 0 or more clauses within the size limits."""
    if rng.random() < spec["empty_share"]:
        return []
    chosen = rng.sample(clauses, rng.randint(1, spec["max_clauses"]))
    picked, chars = [], 0
    for clause in chosen:
        options = pool["clauses"][clause]
        n = 2 if clause in pool["multi"] and len(options) > 1 and rng.random() < 0.3 else 1
        for template in rng.sample(options, n):
            source, answer = slots.fill(template)
            if len(answer) > spec.get("max_passage_chars", len(answer)):
                continue
            if chars + len(answer) > spec.get("max_quote_chars", chars + len(answer)):
                continue
            chars += len(answer)
            picked.append((clause, source, answer))
    return picked


def build_blocks(passages: list[tuple[str, str, str]], pool: dict, clauses: list[str],
                 slots: Slots, rng: random.Random, spec: dict, page: int) -> tuple[list[str], int]:
    """Text blocks to place among the filler. A look-alike can sit right beside its passage."""
    blocks = []
    for _, source, _ in passages:
        if rng.random() < spec["page_break_share"]:
            source, page = break_page(source, page, rng), page + 1
        blocks.append([source])
    near = [c for c, _, _ in passages]
    for _ in range(rng.randint(spec["min_lookalikes"], spec["max_lookalikes"])):
        if near and rng.random() < NEAR_SHARE:
            i = rng.randrange(len(near))
            text = slots.fill(rng.choice(pool["lookalikes"][near[i]]))[0]
            blocks[i].insert(rng.randint(0, len(blocks[i])), text)
        elif rng.random() < 0.5:
            blocks.append([slots.fill(rng.choice(pool["general"]))[0]])
        else:
            blocks.append([slots.fill(rng.choice(pool["lookalikes"][rng.choice(clauses)]))[0]])
    return ["\n\n".join(b) for b in blocks], page


def assemble(blocks: list[str], fill: list[str], rng: random.Random, target: int,
             page: int) -> str:
    """Spread the blocks through the filler and add page numbers between paragraphs."""
    room = max(0, target - sum(len(b) + 2 for b in blocks))
    paragraphs, chars = [], 0
    for p in fill:
        if chars >= room:
            break
        paragraphs.append(p)
        chars += len(p) + 2
    for block in blocks:
        paragraphs.insert(rng.randint(0, len(paragraphs)), block)
    out, since_page = [], 0
    for p in paragraphs:
        out.append(p)
        since_page += len(p)
        if since_page >= PAGE_EVERY_CHARS:
            out.append(str(page))
            page, since_page = page + 1, 0
    return "\n\n".join(out)


def build_example(ex_id: str, pool: dict, groups: list[list[str]], clauses: list[str],
                  rng: random.Random, spec: dict) -> Example:
    slots = Slots(rng)
    page = rng.randint(2, 150)
    passages = pick_passages(pool, clauses, slots, rng, spec)
    blocks, page = build_blocks(passages, pool, clauses, slots, rng, spec, page)
    fill = rng.choice(groups)
    start = rng.randrange(len(fill))
    fill = (fill[start:] + rng.choice(groups) + rng.choice(groups) + rng.choice(groups))
    excerpt = assemble(blocks, fill, rng, rng.randint(spec["min_chars"], spec["max_chars"]), page)
    answer = {c: [a for k, _, a in passages if k == c] for c in clauses}
    gold = {c: [{"text": q, "main": True} for q in qs] for c, qs in answer.items() if qs}
    return {"id": ex_id, "excerpt": excerpt, "gold": gold,
            "answer": json.dumps(answer, ensure_ascii=False, indent=2)}


def build_sets(bank: dict, groups: list[list[str]], spec: dict, seed: int) -> dict[str, list[Example]]:
    """All three sets. Filler for the held-out set comes from contracts the others do not use."""
    clauses = credit_clauses()
    train_pool, held_pool = split_bank(bank)
    cut = len(groups) // 5
    out = {}
    for offset, name in enumerate(SET_NAMES):
        rng = random.Random(seed + offset)
        pool, fill = (held_pool, groups[:cut]) if name == "held_out" else (train_pool, groups[cut:])
        s = dict(spec["defaults"], **spec[name])
        out[name] = [build_example(f"synth-{name}-{i:04d}", pool, fill, clauses, rng, s)
                     for i in range(s["n"])]
    return out


def save_sets(sets: dict[str, list[Example]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in sets.items():
        lines = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples)
        (out_dir / f"{name}.jsonl").write_text(lines, encoding="utf-8")


def load_set(path: Path) -> list[Example]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def leak_check(bank: dict, texts: dict[str, str], words: int = LEAK_WORDS) -> list[str]:
    """Bank wording of `words` words or more that also appears in an agreement."""
    keys = {d: match_key(t) for d, t in texts.items()}
    templates = [t for by in (bank["clauses"], bank["lookalikes"]) for ts in by.values() for t in ts]
    hits = []
    for t in templates + bank["general"]:
        for part in SLOT.split(t):
            run = match_key(part).split()
            for i in range(len(run) - words + 1):
                probe = " ".join(run[i:i + words])
                for d, key in keys.items():
                    if probe in key:
                        hits.append(f"{d}: {probe}")
                        break
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the synthetic excerpts.")
    ap.add_argument("--cuad-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--text-dir", type=Path, help="M1 agreements, for the leak check")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "small_loop.yaml").read_text(encoding="utf-8"))
    bank = load_bank()
    titles = json.loads((REPO_ROOT / "data" / "split_manifest.json").read_text("utf-8"))["train"]
    groups = filler_groups(load_contracts(a.cuad_json), titles)
    sets = build_sets(bank, groups, cfg["synth"], cfg["seed"])
    save_sets(sets, a.out)
    for name, examples in sets.items():
        sizes = sorted(len(e["excerpt"]) for e in examples)
        empty = sum(not e["gold"] for e in examples)
        print(f"{name}: {len(examples)} excerpts, {sizes[0]}-{sizes[-1]} chars, {empty} with no clause")
    if a.text_dir:
        hits = leak_check(bank, {p.stem: p.read_text("utf-8") for p in a.text_dir.glob("*.txt")})
        print(f"leak check: {len(hits)} runs of {LEAK_WORDS} words found in the M1 agreements")
        print("\n".join(hits[:20]))
        return 1 if hits else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
