from openai import OpenAI
from pathlib import Path
import base64, sys, time, os, random

client = OpenAI()
BASE_SLEEP = float(os.getenv("LLM_OCR_SLEEP", "1.0"))
OVERWRITE = os.getenv("OVERWRITE") == "1"
BACKUP = os.getenv("BACKUP") == "1"   # optional: keep old file as .bak

def data_uri(img_path: Path) -> str:
    ext = img_path.suffix.lower()
    mime = "image/png" if ext == ".png" else ("image/jpeg" if ext in (".jpg", ".jpeg") else "image/png")
    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def ocr_image(img_path: Path) -> str:
    backoff = 0.8
    for attempt in range(12):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict OCR engine. "
                            "Output the text EXACTLY as printed, character-for-character. "
                            "DO NOT reflow or wrap text; KEEP EVERY line break as in the image. "
                            "KEEP end-of-line hyphenation. "
                            "Do NOT join words across a line break. "
                            "No additions, no summaries, no guesses. If unreadable, write [UNK]. "
                            "Plain text only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type":"text","text":"Extract ONLY the visible text, preserving line breaks and end-of-line hyphens exactly."},
                            {"type":"image_url","image_url":{"url": data_uri(img_path)}},
                        ],
                    },
                ],
                temperature=0,
                max_tokens=2000,
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            return text
        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg or "limit" in msg:
                time.sleep(backoff + random.uniform(0, 0.5))
                backoff = min(backoff * 1.8, 20.0)
                continue
            if "timeout" in msg or "temporarily unavailable" in msg:
                time.sleep(1.5); continue
            raise
    raise RuntimeError("OCR failed after retries")

def seg_png_to_truth_stem(png_name: str) -> str:
    if "-" in png_name:
        left, right = png_name.rsplit("-", 1)
        right = right.replace(".png", "")
        right = right.zfill(4)[-3:]
        return f"{left}_{right}.txt"
    return png_name.rsplit(".",1)[0] + ".txt"

def main():
    if len(sys.argv) < 3:
        print("Usage: python llm_ocr_segments_page.py <segments_dir> <ocr_llm_output_dir>")
        sys.exit(1)

    seg_dir = Path(sys.argv[1]).expanduser()
    out_dir = Path(sys.argv[2]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(seg_dir.glob("*.png"))
    if not pngs:
        print(f"No PNGs found in {seg_dir}")
        sys.exit(1)

    print(f"OCR-ing {len(pngs)} segments from {seg_dir} -> {out_dir}")
    for i, png in enumerate(pngs, 1):
        out_name = seg_png_to_truth_stem(png.name)
        out_path = out_dir / out_name

        if out_path.exists() and not OVERWRITE:
            print(f"[{i}/{len(pngs)}] Skip (exists) {out_name}")
            continue

        try:
            text = ocr_image(png)
            if out_path.exists() and BACKUP:
                out_path.rename(out_path.with_suffix(out_path.suffix + ".bak"))
            out_path.write_text(text, encoding="utf-8")
            print(f"[{i}/{len(pngs)}] Saved {out_name}{' (overwritten)' if OVERWRITE else ''}")
            time.sleep(BASE_SLEEP)
        except Exception as e:
            print(f"[{i}/{len(pngs)}] ERROR on {png.name}: {e}")

if __name__ == "__main__":
    main()
