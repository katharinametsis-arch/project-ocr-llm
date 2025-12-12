OCR Evaluation (Swedish Newspapers, 1818–1906)

This repo contains a small, reproducible pipeline to:

Extract layout segments from page PDFs

OCR each segment with an LLM (OpenAI Vision)

Score LLM output against the gold transcription (CER/WER)

Compare LLM vs ABBYY (and Tesseract if available)

Generate CSV summaries and a clickable HTML review for qualitative analysis

Data: “svenska-tidningar-1818-1870” / “svenska-tidningar-1871-1906” (KB/Spraakbanken).
Gold (“transcribed”), ABBYY (“ocr_abbyy”), page PDFs (“images_pdf”).


Repository contents

align-mod-clean.py
Minimal evaluator used in early tests (reads two text files, prints CER/WER and word alignment).

extract_segment.py
Splits a page PDF into segment PNGs using the layout boxes embedded in the PDF (via PyMuPDF).

llm_ocr_segments_page.py
Runs OCR for all segment PNGs on a page (OpenAI Vision), writes one .txt per segment.

ocr_eval_tutti.py (formerly ocr_eval_toolkit.py)
Main evaluation toolkit — batch scoring, page summaries, global summaries, qualitative diffs, etc.

create_qual_diff_review.py
Builds a sortable local HTML from the qualitative CSV (clickable links to PNG/TRUTH/LLM/ABBYY).

Requirements

Python 3.9+ (CPython recommended for OpenAI SDK)

Packages:

pip install openai==1.* pymupdf pillow

(If errors with pymupdf, upgrade pip: python -m pip install --upgrade pip)

API key for OpenAI:

export OPENAI_API_KEY="sk-…yourkey…"

Tip (Mac): create a venv just for this work:

python3 -m venv ~/openai-venv
source ~/openai-venv/bin/activate
pip install openai==1.* pymupdf pillow


Directory conventions (assumed)

<DATA_ROOT>/
  svenska-tidningar-1871-1906/
    images_pdf/          # page PDFs (one folder per issue/date)
    transcribed/         # gold, one TXT per segment
    ocr_abbyy/           # ABBYY TXT per segment (if provided)
  segments/              # you will create this: PNGs per page (one dir per page)
  ocr_llm/               # you will create this: LLM OCR TXT per page (one dir per page)

Choose any root; below examples use ~/Downloads/svenska-tidningar-1871-1906 as data root and ~/segments, ~/Downloads/svenska-tidningar-1871-1906/ocr_llm for outputs.


1) Quick start (one page, end-to-end)

# 1. Activate venv and set your key
source ~/openai-venv/bin/activate
export OPENAI_API_KEY="sk-…"

# 2. Extract segments (PNGs) for one PDF page
python extract_segment.py \
  "$HOME/Downloads/svenska-tidningar-1871-1906/images_pdf/SYDSVENSKA DAGBLADET 1887-02-15/bib22651590_18870215_153666_37_0002.Pdf" \
  --outdir "$HOME/segments/bib22651590_18870215_153666_37_0002"

# 3. OCR all PNG segments on that page with the LLM
python llm_ocr_segments_page.py \
  "$HOME/segments/bib22651590_18870215_153666_37_0002" \
  "$HOME/Downloads/svenska-tidningar-1871-1906/ocr_llm/bib22651590_18870215_153666_37_0002"

# 4. Evaluate LLM vs GOLD for that single page (CSV + summary)
python ocr_eval_tutti.py eval-one-page \
  --page-stem "bib22651590_18870215_153666_37_0002" \
  --truth-dir "$HOME/Downloads/svenska-tidningar-1871-1906/transcribed/SYDSVENSKA DAGBLADET 1887-02-15" \
  --llm-dir   "$HOME/Downloads/svenska-tidningar-1871-1906/ocr_llm/bib22651590_18870215_153666_37_0002" \
  --out "$HOME/eval_one_page_llm.csv"

# 5. (Optional) Evaluate ABBYY vs GOLD for the same page
python ocr_eval_tutti.py eval-one-page \
  --page-stem "bib22651590_18870215_153666_37_0002" \
  --truth-dir "$HOME/Downloads/svenska-tidningar-1871-1906/transcribed/SYDSVENSKA DAGBLADET 1887-02-15" \
  --abbyy-dir "$HOME/Downloads/svenska-tidningar-1871-1906/ocr_abbyy/SYDSVENSKA-DAGBLADET-1887-02-15" \
  --out "$HOME/eval_one_page_abbyy.csv"

# 6. Compare LLM vs ABBYY for that page (ΔCER/ΔWER, per-segment wins)
python ocr_eval_tutti.py compare-one-page \
  --page-stem "bib22651590_18870215_153666_37_0002" \
  --truth-dir "$HOME/Downloads/svenska-tidningar-1871-1906/transcribed/SYDSVENSKA DAGBLADET 1887-02-15" \
  --llm-dir   "$HOME/Downloads/svenska-tidningar-1871-1906/ocr_llm/bib22651590_18870215_153666_37_0002" \
  --abbyy-dir "$HOME/Downloads/svenska-tidningar-1871-1906/ocr_abbyy/SYDSVENSKA-DAGBLADET-1887-02-15" \
  --out "$HOME/eval_one_page_compare.csv"


1) Extract all segments (batch)

From all PDFs under images_pdf, produce a segments folder per page:

# Example: loop all PDFs and extract
find "$HOME/Downloads/svenska-tidningar-1871-1906/images_pdf" -type f -iname '*.pdf' | while IFS= read -r pdf; do
  stem="$(basename "${pdf%.*}")"                 # e.g. bib2265…_0002
  out="$HOME/segments/$stem"
  mkdir -p "$out"
  echo "Extracting → $out"
  python extract_segment.py "$pdf" --outdir "$out"
done

extract_segment.py prints the segment rectangles and writes stem-0001.png, stem-0002.png, … into the page’s directory.


2) LLM OCR for all pages (batch)

# Optional: polite throttling to avoid rate limits
export LLM_OCR_SLEEP=0.6   # seconds between calls (tune if needed)

for d in "$HOME/segments"/*; do
  [ -d "$d" ] || continue
  stem="$(basename "$d")"
  out="$HOME/Downloads/svenska-tidningar-1871-1906/ocr_llm/$stem"
  mkdir -p "$out"
  echo "OCR-ing $(ls -1 "$d"/*.png 2>/dev/null | wc -l) segments → $out"
  python llm_ocr_segments_page.py "$d" "$out"
done

The script overwrites existing .txt files by default (good when you fix prompt/line-break policy).

If you hit 429 rate_limit the script backs off and retries. You can resume by running the same command again; processed segments are skipped/overwritten.


3) Evaluate (per-page and totals)

Per-page (LLM or ABBYY vs GOLD)

# LLM vs GOLD across all pages → per-page CSV + per-page summary CSV
python ocr_eval_tutti.py eval-all-llm \
  --truth-root "$HOME/Downloads/svenska-tidningar-1871-1906/transcribed" \
  --llm-root   "$HOME/Downloads/svenska-tidningar-1871-1906/ocr_llm" \
  --out-segments "$HOME/eval_llm_segments.csv" \
  --out-pages    "$HOME/eval_llm_pages.csv"

# ABBYY vs GOLD across all pages
python ocr_eval_tutti.py eval-all-abbyy \
  --truth-root "$HOME/Downloads/svenska-tidningar-1871-1906/transcribed" \
  --abbyy-root "$HOME/Downloads/svenska-tidningar-1871-1906/ocr_abbyy" \
  --out-segments "$HOME/eval_abbyy_segments.csv" \
  --out-pages    "$HOME/eval_abbyy_pages.csv"

These produce:

*_segments.csv — one row per segment (segment id, CER, WER, status)

*_pages.csv — one row per page (avg CER/WER, counts)

Compare LLM vs ABBYY (intersection only)

python ocr_eval_tutti.py compare-pages \
  --llm-pages   "$HOME/eval_llm_pages.csv" \
  --abbyy-pages "$HOME/eval_abbyy_pages.csv" \
  --out         "$HOME/who_wins_by_page.csv"


4) Qualitative samples + review page

Build a CSV of the largest differences

python ocr_eval_tutti.py qual-samples \
  --out "$HOME/qual_diff_samples.csv" \
  --top 200

Turn that CSV into a clickable HTML

python create_qual_diff_review.py \
  --csv "$HOME/qual_diff_samples.csv" \
  --out "$HOME/qual_diff_review.html"

open "$HOME/qual_diff_review.html"

On the page you can filter by Winner, search, sort by ΔCER/ΔWER, and open PNG / TRUTH / LLM / ABBYY with one click.


Metrics explained

CER (Character Error Rate): Levenshtein distance on characters ÷ length of gold text.

WER (Word Error Rate): Levenshtein on whitespace-split tokens ÷ number of gold words.

Both range from 0.0 (perfect) upward. Lower is better.

Normalize Unicode (NFC/NFKC), strip zero-width chars, convert NBSP to space, and optionally remove layout markup (e.g., <b>…</b> or custom tags) before scoring. See ocr_eval_tutti.py.


Troubleshooting

“No PNGs found”: Run extract_segment.py first; point llm_ocr_segments_page.py at that folder.

Rate limits (429): Increase LLM_OCR_SLEEP or re-run later; the script resumes safely.

Quoting paths: Many folder names contain spaces; wrap paths in quotes.

ABBYY folder naming: Some sets use hyphenated names (e.g., SYDSVENSKA-DAGBLADET-1887-02-15). Adjust the --abbyy-dir or --abbyy-root accordingly.


Acknowledgements

Source data courtesy of Kungliga biblioteket (KB) & Språkbanken (Kubhist corpus).
Evaluation framing based on Dannélls et al. (2019), DHN
