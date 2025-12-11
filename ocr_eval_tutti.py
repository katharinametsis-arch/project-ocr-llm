#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Evaluation Toolkit
- Computes CER/WER vs ground truth for LLM and ABBYY outputs (segments + pages)
- Builds system-to-system comparisons on page level (LLM vs ABBYY)
- Produces qualitative samples (side-by-side text) for top diffs
- Character-level Levenshtein breakdown (subs / dels / ins and top confusions)

Paths default to your current project layout but can be overridden with CLI flags.

Python: 3.9+ (standard library only)
"""

from __future__ import annotations
import argparse, csv, re, unicodedata, math
from pathlib import Path
from collections import defaultdict, Counter

# ---------- Defaults (match your tree) ----------
H = Path.home()
TRANS_DEFAULT = H / "Downloads/svenska-tidningar-1871-1906/transcribed"
LLM_DEFAULT   = H / "Downloads/svenska-tidningar-1871-1906/ocr_llm"
ABBYY_DEFAULT = H / "Downloads/svenska-tidningar-1871-1906/ocr_abbyy"
OUT_DEFAULT   = H  # write CSVs here by default

# ---------- Normalization ----------
TAG_RE = re.compile(r"<[^>]+>")
WS_RE  = re.compile(r"\s+")

def norm_truth(t: str) -> str:
    """Strip tags, normalize unicode, remove format chars, fix NBSP/soft hyphen, collapse ws."""
    t = TAG_RE.sub("", t)
    t = unicodedata.normalize("NFKC", t)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf")
    t = t.replace("\u00A0", " ").replace("\u00AD", "")
    t = WS_RE.sub(" ", t).strip()
    return t

def norm_ocr(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf")
    t = t.replace("\u00A0", " ").replace("\u00AD", "")
    t = WS_RE.sub(" ", t).strip()
    return t

# ---------- Utilities ----------
def stem_and_seg(p: Path):
    s = p.stem
    if "_" not in s: return None
    page_stem, seg = s.rsplit("_", 1)
    return page_stem, seg

def find_abbyy_file(abbyy_root: Path, page_stem: str, seg: str) -> Path | None:
    # ABBYY files live under ocr_abbyy/<ISSUE-DASH>/<page_stem>_<seg>.txt OR .Txt
    for ext in (".txt", ".Txt"):
        pattern = str(abbyy_root / "**" / f"{page_stem}_{seg}{ext}")
        hits = list(Path().glob(pattern))  # glob from cwd
        if hits:
            return hits[0]
    # glob() above is from CWD; safer with rglob from root:
    for ext in (".txt", ".Txt"):
        hits = list((abbyy_root).rglob(f"{page_stem}_{seg}{ext}"))
        if hits:
            return hits[0]
    return None

# ---------- Levenshtein (list-based so works for both chars and words) ----------
def levenshtein(a, b):
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            bj = b[j-1]
            cost = 0 if ai == bj else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[n][m]

def cer(ref: str, hyp: str) -> float:
    if not ref: return float("inf")
    return levenshtein(list(ref), list(hyp)) / len(ref)

def wer(ref: str, hyp: str) -> float:
    rw, hw = ref.split(), hyp.split()
    if not rw: return float("inf")
    return levenshtein(rw, hw) / len(rw)

# ---------- Scoring helpers ----------
def iter_truth_segments(trans_root: Path):
    for p in trans_root.rglob("*.txt"):
        ss = stem_and_seg(p)
        if ss:
            yield p, ss[0], ss[1]

def score_system_vs_truth(trans_root: Path, sys_root: Path, out_segments_csv: Path, out_pages_csv: Path):
    """
    Build per-segment (micro) and per-page (macro) CSVs for a system (LLM or ABBYY-like layout).
    - If sys_root is ocr_llm: files at ocr_llm/<page_stem>/<page_stem>_<seg>.txt
    - If sys_root is ocr_abbyy: nested; we try rglob to find each file by name.
    """
    per_page = defaultdict(lambda: {
        "ok": 0, "missing": 0, "empty_ref": 0,
        "sum_cer": 0.0, "sum_wer": 0.0
    })
    seg_rows = []
    n_total = n_ok = 0

    for truth_path, page_stem, seg in iter_truth_segments(trans_root):
        n_total += 1
        ref = norm_truth(truth_path.read_text(encoding="utf-8", errors="ignore"))
        if sys_root.name == "ocr_llm":
            hyp_path = sys_root / page_stem / f"{page_stem}_{seg}.txt"
            if not hyp_path.exists():
                per_page[page_stem]["missing"] += 1
                continue
        else:  # ABBYY layout
            hyp_path = find_abbyy_file(sys_root, page_stem, seg)
            if not hyp_path:
                per_page[page_stem]["missing"] += 1
                continue

        hyp = norm_ocr(hyp_path.read_text(encoding="utf-8", errors="ignore"))
        if not ref:
            per_page[page_stem]["empty_ref"] += 1
            continue

        c = cer(ref, hyp)
        w = wer(ref, hyp)
        seg_rows.append({
            "page_stem": page_stem,
            "segment": seg,
            "CER": round(c, 6),
            "WER": round(w, 6),
            "truth_path": str(truth_path),
            "hyp_path": str(hyp_path),
        })
        per_page[page_stem]["ok"] += 1
        per_page[page_stem]["sum_cer"] += c
        per_page[page_stem]["sum_wer"] += w
        n_ok += 1

    # write segments
    out_segments_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_segments_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["page_stem","segment","CER","WER","truth_path","hyp_path"])
        for r in seg_rows:
            w.writerow([r["page_stem"], r["segment"], f'{r["CER"]:.6f}', f'{r["WER"]:.6f}', r["truth_path"], r["hyp_path"]])

    # write pages
    with out_pages_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["page_stem","segments_ok","segments_missing","segments_empty_ref","avg_CER","avg_WER"])
        for ps, agg in sorted(per_page.items()):
            ok = agg["ok"]
            avg_c = (agg["sum_cer"]/ok) if ok else float("nan")
            avg_w = (agg["sum_wer"]/ok) if ok else float("nan")
            w.writerow([ps, ok, agg["missing"], agg["empty_ref"],
                        f"{avg_c:.4f}" if ok else "", f"{avg_w:.4f}" if ok else ""])

    print(f"Wrote: {out_segments_csv}")
    print(f"Wrote: {out_pages_csv}")
    # quick micro-avg across ok segments
    if seg_rows:
        micro_c = sum(r["CER"] for r in seg_rows)/len(seg_rows)
        micro_w = sum(r["WER"] for r in seg_rows)/len(seg_rows)
        print(f"Segments: {len(seg_rows)}   micro-avg CER: {micro_c:.4f}  WER: {micro_w:.4f}")

# ---------- Compare pages: LLM vs ABBYY ----------
def compare_pages(llm_pages_csv: Path, abbyy_pages_csv: Path, out_by_page: Path,
                  out_total: Path | None = None, out_by_issue: Path | None = None,
                  out_by_year: Path | None = None, trans_root: Path | None = None):
    def load(p: Path):
        d = {}
        with p.open(encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                ps = r["page_stem"]
                d[ps] = {"CER": float(r["avg_CER"]), "WER": float(r["avg_WER"])}
        return d

    L = load(llm_pages_csv)
    A = load(abbyy_pages_csv)
    common = sorted(set(L) & set(A))
    if not common:
        print("No intersection pages; check inputs.")
    # write by_page
    with out_by_page.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["page_stem","cer_llm","cer_abbyy","delta_cer","winner_cer",
                    "wer_llm","wer_abbyy","delta_wer","winner_wer"])
        for ps in common:
            cL, cA = L[ps]["CER"], A[ps]["CER"]
            wL, wA = L[ps]["WER"], A[ps]["WER"]
            dC, dW = cL - cA, wL - wA
            wc = "LLM" if dC < -1e-12 else ("ABBYY" if dC > 1e-12 else "tie")
            ww = "LLM" if dW < -1e-12 else ("ABBYY" if dW > 1e-12 else "tie")
            w.writerow([ps, f"{cL:.6f}", f"{cA:.6f}", f"{dC:.6f}", wc,
                           f"{wL:.6f}", f"{wA:.6f}", f"{dW:.6f}", ww])
    print(f"Wrote: {out_by_page} (pages: {len(common)})")

    # totals
    if out_total:
        rows = []
        with out_by_page.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "dC": float(r["delta_cer"]),
                    "dW": float(r["delta_wer"]),
                    "wC": r["winner_cer"],
                    "wW": r["winner_wer"],
                })
        n = len(rows)
        wcer_llm  = sum(r["wC"]=="LLM"   for r in rows)
        wcer_abby = sum(r["wC"]=="ABBYY" for r in rows)
        wcer_tie  = sum(r["wC"]=="tie"   for r in rows)
        wwer_llm  = sum(r["wW"]=="LLM"   for r in rows)
        wwer_abby = sum(r["wW"]=="ABBYY" for r in rows)
        wwer_tie  = sum(r["wW"]=="tie"   for r in rows)
        mean = lambda xs: (sum(xs)/len(xs) if xs else float("nan"))
        median = lambda xs: (sorted(xs)[len(xs)//2] if xs else float("nan"))
        dCs = [r["dC"] for r in rows]; dWs = [r["dW"] for r in rows]

        with out_total.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pages","cer_LLM_wins","cer_ABBYY_wins","cer_ties",
                        "wer_LLM_wins","wer_ABBYY_wins","wer_ties",
                        "mean_delta_cer","median_delta_cer",
                        "mean_delta_wer","median_delta_wer"])
            w.writerow([n, wcer_llm, wcer_abby, wcer_tie,
                        wwer_llm, wwer_abby, wwer_tie,
                        f"{mean(dCs):.6f}", f"{median(dCs):.6f}",
                        f"{mean(dWs):.6f}", f"{median(dWs):.6f}"])
        print(f"Wrote: {out_total}")

    # grouped by issue & year (if we have transcribed tree to recover names/years)
    if out_by_issue or out_by_year:
        # build map page_stem -> "NEWSPAPER YYYY-MM-DD" (parent dir name in transcribed/)
        name_cache = {}
        def page_to_issue(ps: str) -> str:
            if ps in name_cache: return name_cache[ps]
            hits = list(trans_root.rglob(ps + "_*.txt")) if trans_root else []
            name_cache[ps] = hits[0].parent.name if hits else ""
            return name_cache[ps]

        def page_to_year(ps: str, issue: str) -> str:
            m = re.search(r'_(\d{8})_', ps)
            if m: return m.group(1)[:4]
            m = re.search(r'(\d{4})', issue)
            return m.group(1) if m else ""

        # aggregate
        agg_issue = defaultdict(lambda: {"n":0,"sum_dC":0.0,"sum_dW":0.0,"llmC":0,"abbyC":0,"tieC":0,"llmW":0,"abbyW":0,"tieW":0})
        agg_year  = defaultdict(lambda: {"n":0,"sum_dC":0.0,"sum_dW":0.0,"llmC":0,"abbyC":0,"tieC":0,"llmW":0,"abbyW":0,"tieW":0})

        with out_by_page.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ps = r["page_stem"]
                dC = float(r["delta_cer"]); dW = float(r["delta_wer"])
                wC = r["winner_cer"];       wW = r["winner_wer"]
                issue = page_to_issue(ps);  year = page_to_year(ps, issue)
                for bucket in (agg_issue[issue], agg_year[year]):
                    bucket["n"] += 1
                    bucket["sum_dC"] += dC
                    bucket["sum_dW"] += dW
                    if wC=="LLM": bucket["llmC"]+=1
                    elif wC=="ABBYY": bucket["abbyC"]+=1
                    else: bucket["tieC"]+=1
                    if wW=="LLM": bucket["llmW"]+=1
                    elif wW=="ABBYY": bucket["abbyW"]+=1
                    else: bucket["tieW"]+=1

        def write_group(outp: Path, rows, label: str):
            with outp.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow([label,"n_pages","mean_delta_cer","llm_better_cer","abbyy_better_cer","ties_cer",
                            "mean_delta_wer","llm_better_wer","abbyy_better_wer","ties_wer"])
                for k, b in sorted(rows.items()):
                    if not k: continue
                    w.writerow([k, b["n"],
                                f"{(b['sum_dC']/b['n']):.6f}", b["llmC"], b["abbyC"], b["tieC"],
                                f"{(b['sum_dW']/b['n']):.6f}", b["llmW"], b["abbyW"], b["tieW"]])
            print(f"Wrote: {outp}")

        if out_by_issue: write_group(out_by_issue, agg_issue, "issue")
        if out_by_year:  write_group(out_by_year,  agg_year,  "year")

# ---------- Qualitative diffs (per segment) ----------
def qual_samples(trans_root: Path, llm_root: Path, abbyy_root: Path,
                 out_csv: Path, top_per_side: int = 20):
    """
    Pick segments (intersection) where |CER_LLM - CER_ABBYY| is largest.
    Write PNG + text paths for manual inspection.
    """
    rows = []
    # Map truth segs -> paths
    truth_map = {}
    for p in trans_root.rglob("*.txt"):
        ss = stem_and_seg(p)
        if not ss: continue
        page_stem, seg = ss
        truth_map[(page_stem, seg)] = p

    def load_sys(sys_root: Path) -> dict:
        d = {}
        for (page_stem, seg), tpath in truth_map.items():
            if sys_root.name == "ocr_llm":
                hpath = sys_root / page_stem / f"{page_stem}_{seg}.txt"
            else:
                hpath = find_abbyy_file(sys_root, page_stem, seg)
            if hpath and hpath.exists():
                d[(page_stem, seg)] = hpath
        return d

    L = load_sys(llm_root)
    A = load_sys(abbyy_root)
    common = sorted(set(L) & set(A) & set(truth_map))
    for key in common:
        page_stem, seg = key
        ref = norm_truth(truth_map[key].read_text(encoding="utf-8", errors="ignore"))
        hypL = norm_ocr(L[key].read_text(encoding="utf-8", errors="ignore"))
        hypA = norm_ocr(A[key].read_text(encoding="utf-8", errors="ignore"))
        cL, cA = cer(ref, hypL), cer(ref, hypA)
        rows.append((abs(cL-cA), cL, cA, page_stem, seg))

    rows.sort(reverse=True)  # biggest diffs first
    pick = rows[:top_per_side*2]  # over-sample, we’ll balance sides below

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side","delta_abs_CER","page_stem","segment","PNG","TRUTH_TXT","LLM_TXT","ABBYY_TXT"])
        llm_better = 0
        abby_better = 0
        for diff, cL, cA, ps, sg in pick:
            side = "LLM_better" if (cL < cA) else "ABBYY_better"
            # try to locate the PNG under ~/segments/<page_stem>/<page_stem>-NNNN.png
            png = (H/"segments"/ps/f"{ps}-{int(sg):04d}.png")
            w.writerow([
                side, f"{diff:.6f}", ps, sg,
                str(png),
                str(truth_map[(ps, sg)]),
                str(LLM_ROOT(ps)/f"{ps}_{sg}.txt") if (LLM_ROOT(ps)/f"{ps}_{sg}.txt").exists() else "",
                str(find_abbyy_file(abbyy_root, ps, sg) or "")
            ])
            if side == "LLM_better": llm_better += 1
            else: abby_better += 1

    print(f"Wrote: {out_csv}  (approx top {top_per_side} each side; actual depends on diffs)")

def LLM_ROOT(ps: str) -> Path:
    return LLM_DEFAULT / ps  # helper for printing path

# ---------- Levenshtein character ops totals ----------
def lev_char_breakdown(trans_root: Path, llm_root: Path, abbyy_root: Path, out_dir: Path):
    def lev_ops(ref: str, hyp: str):
        n, m = len(ref), len(hyp)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1): dp[i][0] = i
        for j in range(m+1): dp[0][j] = j
        for i in range(1, n+1):
            ri = ref[i-1]
            for j in range(1, m+1):
                hj = hyp[j-1]
                cost = 0 if ri==hj else 1
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
        i, j = n, m
        subs = dels = ins = 0
        sub_pairs = Counter(); del_chars = Counter(); ins_chars = Counter()
        while i>0 or j>0:
            if i>0 and j>0 and dp[i][j] == dp[i-1][j-1] + (ref[i-1]!=hyp[j-1]):
                if ref[i-1] != hyp[j-1]:
                    subs += 1; sub_pairs[(ref[i-1], hyp[j-1])] += 1
                i -= 1; j -= 1
            elif i>0 and dp[i][j] == dp[i-1][j] + 1:
                dels += 1; del_chars[ref[i-1]] += 1; i -= 1
            else:
                ins += 1; ins_chars[hyp[j-1]] += 1; j -= 1
        return subs, dels, ins, sub_pairs, del_chars, ins_chars, n

    # Build intersection
    truth_map = {}
    for p in trans_root.rglob("*.txt"):
        ss = stem_and_seg(p)
        if ss: truth_map[ss] = p

    def hyp_path(sys_root: Path, page_stem: str, seg: str) -> Path | None:
        if sys_root.name == "ocr_llm":
            p = sys_root / page_stem / f"{page_stem}_{seg}.txt"
            return p if p.exists() else None
        return find_abbyy_file(sys_root, page_stem, seg)

    totals = {
        "LLM":  {"subs":0,"dels":0,"ins":0,"ref_len":0,
                 "sub_pairs":Counter(),"del_chars":Counter(),"ins_chars":Counter()},
        "ABBYY":{"subs":0,"dels":0,"ins":0,"ref_len":0,
                 "sub_pairs":Counter(),"del_chars":Counter(),"ins_chars":Counter()},
    }
    inter = 0
    for (page_stem, seg), tpath in truth_map.items():
        lp = hyp_path(llm_root, page_stem, seg)
        ap = hyp_path(abbyy_root, page_stem, seg)
        if not (lp and ap): continue
        ref = norm_truth(tpath.read_text(encoding="utf-8", errors="ignore"))
        hypL = norm_ocr(lp.read_text(encoding="utf-8", errors="ignore"))
        hypA = norm_ocr(ap.read_text(encoding="utf-8", errors="ignore"))
        for name, hyp in (("LLM", hypL), ("ABBYY", hypA)):
            s,d,i, sp, dc, ic, n = lev_ops(ref, hyp)
            T = totals[name]
            T["subs"]+=s; T["dels"]+=d; T["ins"]+=i; T["ref_len"]+=n
            T["sub_pairs"].update(sp); T["del_chars"].update(dc); T["ins_chars"].update(ic)
        inter += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    # totals file
    with (out_dir/"lev_char_ops_totals.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system","ref_chars","edits","CER","subs","dels","ins",
                    "subs_pct","dels_pct","ins_pct"])
    for sys in ("LLM","ABBYY"):
        T = totals[sys]
        edits = T["subs"] + T["dels"] + T["ins"]
        CER = edits / T["ref_len"] if T["ref_len"] else float("nan")
        with (out_dir/"lev_char_ops_totals.csv").open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([sys, T["ref_len"], edits, f"{CER:.6f}",
                        T["subs"], T["dels"], T["ins"],
                        f"{(T['subs']/edits if edits else 0):.6f}",
                        f"{(T['dels']/edits if edits else 0):.6f}",
                        f"{(T['ins']/edits if edits else 0):.6f}"])
    # tops
    def write_pairs(counter: Counter, outp: Path):
        with outp.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["ref_char","hyp_char","count"])
            for (a,b), c in counter.most_common(200):
                w.writerow([a,b,c])
    write_pairs(totals["LLM"]["sub_pairs"],   out_dir/"lev_top_substitutions_llm.csv")
    write_pairs(totals["ABBYY"]["sub_pairs"], out_dir/"lev_top_substitutions_abbyy.csv")
    def write_one(counter: Counter, outp: Path, header: str):
        with outp.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow([header,"count"])
            for ch, c in counter.most_common(200): w.writerow([ch, c])
    write_one(totals["LLM"]["ins_chars"],   out_dir/"lev_top_insertions_llm.csv",   "inserted_char")
    write_one(totals["LLM"]["del_chars"],   out_dir/"lev_top_deletions_llm.csv",    "deleted_char")
    write_one(totals["ABBYY"]["ins_chars"], out_dir/"lev_top_insertions_abbyy.csv", "inserted_char")
    write_one(totals["ABBYY"]["del_chars"], out_dir/"lev_top_deletions_abbyy.csv",  "deleted_char")
    print(f"Intersection segments: {inter}")
    print("Wrote:", out_dir/"lev_char_ops_totals.csv")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="OCR Evaluation Toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # eval-llm
    s1 = sub.add_parser("eval-llm", help="Score LLM vs truth (segments & pages CSV).")
    s1.add_argument("--trans", type=Path, default=TRANS_DEFAULT)
    s1.add_argument("--llm",   type=Path, default=LLM_DEFAULT)
    s1.add_argument("--outdir",type=Path, default=OUT_DEFAULT)

    # eval-abbyy
    s2 = sub.add_parser("eval-abbyy", help="Score ABBYY vs truth (segments & pages CSV).")
    s2.add_argument("--trans", type=Path, default=TRANS_DEFAULT)
    s2.add_argument("--abbyy", type=Path, default=ABBYY_DEFAULT)
    s2.add_argument("--outdir",type=Path, default=OUT_DEFAULT)

    # compare-pages
    s3 = sub.add_parser("compare-pages", help="Compare LLM vs ABBYY on page averages.")
    s3.add_argument("--llm-pages",   type=Path, default=H/"eval_llm_pages.csv")
    s3.add_argument("--abbyy-pages", type=Path, default=H/"eval_abbyy_pages.csv")
    s3.add_argument("--outdir",      type=Path, default=OUT_DEFAULT)
    s3.add_argument("--trans",       type=Path, default=TRANS_DEFAULT)

    # qual-samples
    s4 = sub.add_parser("qual-samples", help="Emit qualitative diff samples as CSV.")
    s4.add_argument("--trans", type=Path, default=TRANS_DEFAULT)
    s4.add_argument("--llm",   type=Path, default=LLM_DEFAULT)
    s4.add_argument("--abbyy", type=Path, default=ABBYY_DEFAULT)
    s4.add_argument("--out",   type=Path, default=H/"qual_diff_samples.csv")
    s4.add_argument("--top",   type=int, default=20)

    # lev-breakdown
    s5 = sub.add_parser("lev-breakdown", help="Character-level Levenshtein breakdown (totals + top confusions).")
    s5.add_argument("--trans", type=Path, default=TRANS_DEFAULT)
    s5.add_argument("--llm",   type=Path, default=LLM_DEFAULT)
    s5.add_argument("--abbyy", type=Path, default=ABBYY_DEFAULT)
    s5.add_argument("--outdir",type=Path, default=OUT_DEFAULT)

    args = ap.parse_args()

    if args.cmd == "eval-llm":
        out_segments = args.outdir / "eval_llm_segments.csv"
        out_pages    = args.outdir / "eval_llm_pages.csv"
        score_system_vs_truth(args.trans, args.llm, out_segments, out_pages)

    elif args.cmd == "eval-abbyy":
        out_segments = args.outdir / "eval_abbyy_segments.csv"
        out_pages    = args.outdir / "eval_abbyy_pages.csv"
        score_system_vs_truth(args.trans, args.abbyy, out_segments, out_pages)

    elif args.cmd == "compare-pages":
        out_by_page  = args.outdir / "who_wins_by_page.csv"
        out_total    = args.outdir / "who_wins_TOTAL.csv"
        out_by_issue = args.outdir / "who_wins_by_issue.csv"
        out_by_year  = args.outdir / "who_wins_by_year.csv"
        compare_pages(args.llm_pages, args.abbyy_pages, out_by_page,
                      out_total, out_by_issue, out_by_year, args.trans)

    elif args.cmd == "qual-samples":
        qual_samples(args.trans, args.llm, args.abbyy, args.out, args.top)

    elif args.cmd == "lev-breakdown":
        lev_char_breakdown(args.trans, args.llm, args.abbyy, args.outdir)

if __name__ == "__main__":
    main()
