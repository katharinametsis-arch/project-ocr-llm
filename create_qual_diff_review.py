#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a local, sortable HTML review page from qual_diff_samples.csv

Expected CSV columns:
page_stem,segment,png_path,truth_path,llm_path,abbyy_path,
cer_llm,cer_abbyy,wer_llm,wer_abbyy,delta_cer,delta_wer

Usage:
  python create_qual_diff_review.py --csv "$HOME/qual_diff_samples.csv" --out "$HOME/qual_diff_review.html"
"""
import argparse, csv, html, os, sys
from pathlib import Path
from urllib.parse import quote

def file_url(p: str) -> str:
    p = os.path.abspath(p)
    return "file://" + quote(p)

def parse_float(s):
    try:
        return float(s)
    except Exception:
        return None

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Qualitative OCR Comparison — LLM vs ABBYY</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#0b0e14; --fg:#e6e1cf; --muted:#a1a1a1; --accent:#6cb6ff; --win:#2ecc71; --lose:#ff7675; --tie:#ffd166; }
  body { background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin:0; }
  header { padding:16px 20px; position:sticky; top:0; background:rgba(11,14,20,0.9); backdrop-filter: blur(6px); border-bottom: 1px solid #1f2430; z-index:10; }
  h1 { font-size:18px; margin:0 0 6px 0; }
  .controls { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
  input[type="search"], select { background:#11151d; color:var(--fg); border:1px solid #2a303c; border-radius:8px; padding:8px 10px; }
  .pill { padding:6px 10px; border-radius:999px; border:1px solid #2a303c; background:#11151d; cursor:pointer; user-select:none; }
  .pill.active { border-color: var(--accent); }
  main { padding:16px 20px 40px; }
  table { width:100%; border-collapse:collapse; }
  th, td { padding:10px 10px; border-bottom:1px solid #1f2430; vertical-align:top; }
  th { text-align:left; position:sticky; top:64px; background:#0b0e14; border-bottom:2px solid #2a303c; cursor:pointer; }
  th.sort-asc::after { content:" ▲"; color:var(--muted); }
  th.sort-desc::after { content:" ▼"; color:var(--muted); }
  tr:hover { background:#0f131b; }
  code { background:#11151d; padding:2px 6px; border-radius:6px; border:1px solid #2a303c; }
  .win { color:var(--win); font-weight:600; }
  .lose { color:var(--lose); font-weight:600; }
  .tie { color:var(--tie); font-weight:600; }
  a { color:var(--accent); text-decoration:none; }
  a:hover { text-decoration:underline; }
  .muted { color:var(--muted); }
  .nowrap { white-space:nowrap; }
</style>
</head>
<body>

<header>
  <h1>Qualitative OCR Comparison — LLM vs ABBYY</h1>
  <div class="controls">
    <input id="searchBox" type="search" placeholder="Search page, segment or text paths…" style="min-width:260px">
    <label class="muted">Winner:</label>
    <select id="winnerFilter">
      <option value="">All</option>
      <option value="LLM">LLM</option>
      <option value="ABBYY">ABBYY</option>
      <option value="Tie">Tie</option>
    </select>
    <label class="muted">Sort by:</label>
    <select id="sortKey">
      <option value="abs_delta">|ΔCER| (abs)</option>
      <option value="delta_cer">ΔCER</option>
      <option value="delta_wer">ΔWER</option>
      <option value="cer_llm">CER LLM</option>
      <option value="cer_abbyy">CER ABBYY</option>
      <option value="wer_llm">WER LLM</option>
      <option value="wer_abbyy">WER ABBYY</option>
      <option value="page_stem">Page</option>
      <option value="segment">Segment</option>
    </select>
    <select id="sortDir">
      <option value="desc">Desc</option>
      <option value="asc">Asc</option>
    </select>
    <span class="pill" id="top10">Top 10</span>
    <span class="pill" id="top50">Top 50</span>
    <span class="pill" id="topAll">All</span>
    <span class="muted" id="countInfo"></span>
  </div>
</header>

<main>
  <table id="tbl">
    <thead>
      <tr>
        <th data-key="page_stem">Page</th>
        <th data-key="segment">Seg</th>
        <th data-key="winner">Winner</th>
        <th data-key="delta_cer">ΔCER</th>
        <th data-key="delta_wer">ΔWER</th>
        <th data-key="cer_llm">CER LLM</th>
        <th data-key="cer_abbyy">CER ABBYY</th>
        <th data-key="wer_llm">WER LLM</th>
        <th data-key="wer_abbyy">WER ABBYY</th>
        <th>PNG</th>
        <th>TRUTH</th>
        <th>LLM</th>
        <th>ABBYY</th>
      </tr>
    </thead>
    <tbody id="rows">
      {ROWS}
    </tbody>
  </table>
</main>

<script>
(function(){
  const winnerFilter = document.getElementById('winnerFilter');
  const searchBox = document.getElementById('searchBox');
  const sortKey = document.getElementById('sortKey');
  const sortDir = document.getElementById('sortDir');
  const rowsTbody = document.getElementById('rows');
  const countInfo = document.getElementById('countInfo');
  const top10 = document.getElementById('top10');
  const top50 = document.getElementById('top50');
  const topAll = document.getElementById('topAll');

  const DATA = Array.from(rowsTbody.querySelectorAll('tr')).map(tr => {
    const obj = {};
    for (const td of tr.children) {
      const k = td.getAttribute('data-k');
      if (k) obj[k] = td.getAttribute('data-v') ?? td.textContent.trim();
    }
    obj._el = tr;
    obj.abs_delta = Math.abs(parseFloat(obj.delta_cer || '0')) || 0;
    return obj;
  });

  let limit = Infinity;

  function apply(){
    const q = (searchBox.value||'').toLowerCase();
    const wf = winnerFilter.value;
    const sk = sortKey.value;
    const sd = sortDir.value;

    let filtered = DATA.filter(o => {
      if (wf && o.winner !== wf) return false;
      if (q) {
        const hay = (o.page_stem + " " + o.segment + " " + (o.png_path||"") + " " + (o.truth_path||"") + " " + (o.llm_path||"") + " " + (o.abbyy_path||"")).toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });

    filtered.sort((a,b)=>{
      let va = a[sk], vb = b[sk];
      const na = parseFloat(va), nb = parseFloat(vb);
      if (!Number.isNaN(na) && !Number.isNaN(nb)) { va = na; vb = nb; }
      if (va < vb) return (sd === 'asc') ? -1 : 1;
      if (va > vb) return (sd === 'asc') ? 1 : -1;
      return 0;
    });

    const toShow = filtered.slice(0, limit);

    for (const o of DATA) o._el.style.display = 'none';
    for (const o of toShow) o._el.style.display = '';
    countInfo.textContent = `Showing ${toShow.length} / ${filtered.length} (total ${DATA.length})`;
  }

  winnerFilter.onchange = apply;
  searchBox.oninput = apply;
  sortKey.onchange = apply;
  sortDir.onchange = apply;
  top10.onclick = ()=>{ limit=10; top10.classList.add('active'); top50.classList.remove('active'); topAll.classList.remove('active'); apply(); };
  top50.onclick = ()=>{ limit=50; top50.classList.add('active'); top10.classList.remove('active'); topAll.classList.remove('active'); apply(); };
  topAll.onclick = ()=>{ limit=Infinity; topAll.classList.add('active'); top10.classList.remove('active'); top50.classList.remove('active'); apply(); };

  topAll.classList.add('active');
  apply();
})();
</script>
</body>
</html>
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to qual_diff_samples.csv")
    ap.add_argument("--out", required=True, help="Output HTML path, e.g. ~/qual_diff_review.html")
    args = ap.parse_args()

    csv_path = Path(os.path.expanduser(args.csv))
    out_path = Path(os.path.expanduser(args.out))

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows_html = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            page = r.get("page_stem","")
            seg  = r.get("segment","")
            png  = r.get("png_path","")
            trt  = r.get("truth_path","")
            llm  = r.get("llm_path","")
            abby = r.get("abbyy_path","")

            cer_l = parse_float(r.get("cer_llm"))
            cer_a = parse_float(r.get("cer_abbyy"))
            wer_l = parse_float(r.get("wer_llm"))
            wer_a = parse_float(r.get("wer_abbyy"))
            dcer  = parse_float(r.get("delta_cer"))
            dwer  = parse_float(r.get("delta_wer"))

            if dcer is None and (cer_l is not None and cer_a is not None):
                dcer = (cer_l - cer_a)
            if dwer is None and (wer_l is not None and wer_a is not None):
                dwer = (wer_l - wer_a)
            if cer_l is not None and cer_a is not None:
                if abs(cer_l - cer_a) < 1e-12:
                    winner = "Tie"
                else:
                    winner = "LLM" if cer_l < cer_a else "ABBYY"
            else:
                winner = ""

            page_h = html.escape(page)
            seg_h  = html.escape(seg)
            winner_h = html.escape(winner)
            cls = "tie"
            if winner == "LLM": cls = "win"
            elif winner == "ABBYY": cls = "lose"

            def fmt(x):
                return "" if x is None else f"{x:.4f}"

            def link_cell(path, label):
                if not path: return f'<td data-k="{label}_path" data-v=""></td>'
                return f'<td data-k="{label}_path" data-v="{html.escape(path)}"><a href="{file_url(path)}" target="_blank">open</a></td>'

            rows_html.append(f"""
<tr>
  <td data-k="page_stem" data-v="{page_h}"><code>{page_h}</code></td>
  <td data-k="segment" data-v="{seg_h}" class="nowrap">{seg_h}</td>
  <td data-k="winner" data-v="{winner_h}" class="{cls}">{winner_h}</td>
  <td data-k="delta_cer" data-v="{fmt(dcer)}">{fmt(dcer)}</td>
  <td data-k="delta_wer" data-v="{fmt(dwer)}">{fmt(dwer)}</td>
  <td data-k="cer_llm" data-v="{fmt(cer_l)}">{fmt(cer_l)}</td>
  <td data-k="cer_abbyy" data-v="{fmt(cer_a)}">{fmt(cer_a)}</td>
  <td data-k="wer_llm" data-v="{fmt(wer_l)}">{fmt(wer_l)}</td>
  <td data-k="wer_abbyy" data-v="{fmt(wer_a)}">{fmt(wer_a)}</td>
  {link_cell(png, "png")}
  {link_cell(trt, "truth")}
  {link_cell(llm, "llm")}
  {link_cell(abby, "abbyy")}
</tr>""")

    html_out = TEMPLATE.replace("{ROWS}", "\n".join(rows_html))
    out_path.write_text(html_out, encoding="utf-8")
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
