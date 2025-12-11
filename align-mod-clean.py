#!/usr/bin/env pypy3
# -*- coding: utf-8 -*-

import sys

def levenshtein_matrix(ref, hyp):
  """Compute Levenshtein distance matrix for alignment."""
  n, m = len(ref), len(hyp)
  dp = [[0] * (m + 1) for _ in range(n + 1)]

  for i in range(n + 1):
    dp[i][0] = i
  for j in range(m + 1):
    dp[0][j] = j

  for i in range(1, n + 1):
    for j in range(1, m + 1):
      cost = 0 if ref[i - 1] == hyp[j - 1] else 1
      dp[i][j] = min(
        dp[i - 1][j] + 1,   # deletion
        dp[i][j - 1] + 1,   # insertion
        dp[i - 1][j - 1] + cost # substitution/match
      )
  return dp

def backtrace_alignment(ref, hyp, dp):
  """Reconstruct alignment from the Levenshtein matrix."""
  i, j = len(ref), len(hyp)
  aligned_ref, aligned_hyp = [], []

  while i > 0 or j > 0:
    if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
      aligned_ref.append(ref[i - 1])
      aligned_hyp.append(hyp[j - 1])
      i, j = i - 1, j - 1
    elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
      aligned_ref.append(ref[i - 1])
      aligned_hyp.append("∅") # deletion
      i -= 1
    else:
      aligned_ref.append("∅") # insertion
      aligned_hyp.append(hyp[j - 1])
      j -= 1

  return list(reversed(aligned_ref)), list(reversed(aligned_hyp))

def levenshtein(a, b):
  """Compute Levenshtein distance only."""
  dp = levenshtein_matrix(a, b)
  return dp[len(a)][len(b)]

def cer(ref, hyp):
  dist = levenshtein(ref, hyp)
  return dist / len(ref) if len(ref) > 0 else float('inf')

def wer(ref, hyp):
  ref_words = ref.split()
  hyp_words = hyp.split()
  dist = levenshtein(ref_words, hyp_words)
  return dist / len(ref_words) if len(ref_words) > 0 else float('inf')

def word_alignment(ref, hyp):
  """Return alignment of words with errors marked."""
  ref_words = ref.split()
  hyp_words = hyp.split()
  dp = levenshtein_matrix(ref_words, hyp_words)
  aligned_ref, aligned_hyp = backtrace_alignment(ref_words, hyp_words, dp)
  return aligned_ref, aligned_hyp

def main():
  if len(sys.argv) < 3:
    print("Usage: script.py TRUTH_FILE OCR_FILE")
    sys.exit(1)

  truth_file = sys.argv[1]
  ocr_file = sys.argv[2]

  with open(truth_file, encoding="utf-8") as f:
    truth_text = f.read().strip()
  with open(ocr_file, encoding="utf-8") as f:
    ocr_text = f.read().strip()

  # Metrics
  cer_value = cer(truth_text, ocr_text)
  wer_value = wer(truth_text, ocr_text)

  print(f"Character Error Rate (CER): {cer_value:.4f}")
  print(f"Word Error Rate (WER): {wer_value:.4f}\n")

  # Alignment
  aligned_ref, aligned_hyp = word_alignment(truth_text, ocr_text)
  print("Word alignment (∅ = missing word):")
  for r, h in zip(aligned_ref, aligned_hyp):
    marker = "✓" if r == h else "✗"
    print(f"{r:15s} | {h:15s} {marker}")

if __name__ == "__main__":
  main()
