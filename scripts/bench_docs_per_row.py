"""Measure docs-per-row distribution in the BOS-bestfit dataloader.

Tells us the headroom for cross-doc attention masking:
- If most rows contain 1-2 docs, cross-doc leak is small, masking gains are ~0.
- If rows routinely pack 5+ short docs, cross-doc contamination is a real training-signal
  bug and the ~80 LOC FA3 varlen refactor is worth it.

Also reports cropping rate (% of tokens discarded), which bounds the separate
"token recovery" intervention.

Runs single-process (no DDP); reads from the same parquet shards as training.
Tokenization is on CPU, so this is fine without GPU.

Usage on CKS via BENCH_MODE:
    BENCH_MODE=true BENCH_CMD=scripts.bench_docs_per_row
"""

import argparse
import sys
from collections import Counter

import pyarrow.parquet as pq

from nanochat.tokenizer import get_tokenizer
from nanochat.dataset import list_parquet_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096, help="rows to pack (each row = one (T+1)-token training sequence)")
    parser.add_argument("--seq-len", type=int, default=2048, help="training seq len T; row capacity is T+1")
    parser.add_argument("--buffer-size", type=int, default=1000, help="matches dataloader buffer_size")
    parser.add_argument("--shard-index", type=int, default=0, help="which parquet shard to sample from")
    args = parser.parse_args()

    tokenizer = get_tokenizer()
    bos = tokenizer.get_bos_token_id()
    print(f"BOS token id: {bos}")

    parquet_paths = list_parquet_files()
    assert args.shard_index < len(parquet_paths) - 1, "keep last shard as val"
    shard = parquet_paths[args.shard_index]
    print(f"Reading shard: {shard}")

    pf = pq.ParquetFile(shard)
    print(f"row_groups={pf.num_row_groups}")

    # Replicate the BOS-bestfit packing loop verbatim (see nanochat/dataloader.py).
    row_capacity = args.seq_len + 1
    doc_buffer = []  # list[list[int]] of tokenized docs (each prepended with BOS)

    rg_idx = 0
    def refill_to(n):
        nonlocal rg_idx
        while len(doc_buffer) < n:
            if rg_idx >= pf.num_row_groups:
                return False
            rg = pf.read_row_group(rg_idx)
            batch = rg.column("text").to_pylist()
            for i in range(0, len(batch), 128):
                chunk = batch[i:i + 128]
                token_lists = tokenizer.encode(chunk, prepend=bos, num_threads=4)
                for tokens in token_lists:
                    doc_buffer.append(tokens)
            rg_idx += 1
        return True

    # Stats
    docs_per_row = []
    crop_tokens_per_row = []
    bos_per_row = []
    rows_built = 0
    total_tokens = 0
    total_cropped = 0
    doc_lens = []

    while rows_built < args.rows:
        if not refill_to(args.buffer_size):
            print(f"Exhausted shard after {rows_built} rows", file=sys.stderr)
            break

        pos = 0
        docs_in_this_row = 0
        bos_in_this_row = 0
        cropped_this_row = 0
        while pos < row_capacity:
            # Ensure buffer refilled when it shrinks below threshold (match dataloader).
            if len(doc_buffer) < args.buffer_size:
                if not refill_to(args.buffer_size):
                    break

            remaining = row_capacity - pos
            best_idx, best_len = -1, 0
            for i, d in enumerate(doc_buffer):
                L = len(d)
                if L <= remaining and L > best_len:
                    best_idx, best_len = i, L

            if best_idx >= 0:
                d = doc_buffer.pop(best_idx)
                # d[0] is the BOS prepended by tokenizer.encode(prepend=bos)
                bos_in_this_row += sum(1 for tok in d if tok == bos)
                docs_in_this_row += 1
                doc_lens.append(len(d))
                pos += len(d)
            else:
                shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                d = doc_buffer.pop(shortest_idx)
                crop = min(remaining, len(d))
                cropped_this_row += len(d) - crop  # discarded tail of cropped doc
                bos_in_this_row += 1 if d[0] == bos else 0
                docs_in_this_row += 1
                doc_lens.append(len(d))
                pos += crop

        rows_built += 1
        docs_per_row.append(docs_in_this_row)
        bos_per_row.append(bos_in_this_row)
        crop_tokens_per_row.append(cropped_this_row)
        total_tokens += pos
        total_cropped += cropped_this_row

    n = len(docs_per_row)
    print(f"\n=== Measurement over {n} rows (seq_len={args.seq_len}) ===\n")

    def pct(x, d):
        return 100.0 * x / d if d else float("nan")

    def summ(label, arr):
        arr_sorted = sorted(arr)
        mean = sum(arr) / len(arr)
        p50 = arr_sorted[len(arr_sorted) // 2]
        p90 = arr_sorted[int(len(arr_sorted) * 0.9)]
        p99 = arr_sorted[int(len(arr_sorted) * 0.99)]
        mx = arr_sorted[-1]
        print(f"{label}: mean={mean:.2f}  p50={p50}  p90={p90}  p99={p99}  max={mx}")

    summ("docs_per_row", docs_per_row)
    summ("bos_per_row ", bos_per_row)
    summ("crop_tokens ", crop_tokens_per_row)

    print(f"\nCrop rate: {pct(total_cropped, total_tokens):.2f}% tokens discarded "
          f"(docstring claim: ~35% at T=2048; we only count end-of-row crops, so this is a lower bound)")

    print(f"\nDocs-per-row histogram (bins of 1, truncated at 15):")
    hist = Counter(min(x, 15) for x in docs_per_row)
    for k in sorted(hist):
        bar = "#" * (hist[k] * 60 // max(hist.values()))
        label = f"{k}+" if k == 15 else str(k)
        print(f"  {label:>3} docs: {hist[k]:>5}  {bar}")

    # Doc length summary
    dl_sorted = sorted(doc_lens)
    print(f"\nDoc length (tokens): n={len(dl_sorted)}  mean={sum(doc_lens)/len(doc_lens):.0f}  "
          f"p50={dl_sorted[len(dl_sorted)//2]}  p90={dl_sorted[int(len(dl_sorted)*0.9)]}  "
          f"p99={dl_sorted[int(len(dl_sorted)*0.99)]}  max={dl_sorted[-1]}")

    # Interpretation
    median = sorted(docs_per_row)[len(docs_per_row) // 2]
    print(f"\n=== Interpretation ===")
    if median <= 2:
        print(f"median={median}: cross-doc masking has LOW headroom. Most rows are ≤2 docs, so")
        print("the causal mask already sees mostly-continuous document context.")
        print("=> recommend NOT building BFD varlen masking; look at token-recovery or FP8 rowwise instead.")
    elif median <= 4:
        print(f"median={median}: cross-doc masking has MODERATE headroom. Worth a 2000-iter A/B test")
        print("before committing to the full varlen refactor.")
    else:
        print(f"median={median}: cross-doc masking has HIGH headroom. Real cross-doc contamination in")
        print("training signal. Worth the ~80 LOC FA3 varlen refactor; expect +0.3-0.5 CORE.")


if __name__ == "__main__":
    main()
