"""
Stream HuggingFaceTB/cosmopedia-v2 and write ~24 parquet shards (~6B chars total)
matching the ClimbMix shard schema (single 'text' column, ~250MB compressed each,
row_group_size=1024). Output goes to $NANOCHAT_BASE_DIR/base_data_cosmopedia_v2/
so it can be selected at runtime via NANOCHAT_DATASET_DIR=base_data_cosmopedia_v2.

Run as a one-off job inside the H100 pod — needs HF download bandwidth and disk.
"""
import os
import time
import argparse

from datasets import load_dataset
import pyarrow.parquet as pq
import pyarrow as pa

parser = argparse.ArgumentParser()
parser.add_argument("--num-shards", type=int, default=24, help="number of output shards")
parser.add_argument("--chars-per-shard", type=int, default=250_000_000)
parser.add_argument("--row-group-size", type=int, default=1024)
parser.add_argument("--output-dirname", type=str, default="base_data_cosmopedia_v2")
parser.add_argument("--shuffle-buffer", type=int, default=10_000, help="streaming shuffle buffer size; 0 = no shuffle")
args = parser.parse_args()

base_dir = os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))
output_dir = os.path.join(base_dir, args.output_dirname)
os.makedirs(output_dir, exist_ok=True)
print(f"Output: {output_dir}")
print(f"Target: {args.num_shards} shards x {args.chars_per_shard:,} chars = {args.num_shards * args.chars_per_shard:,} total chars")

# Stream the dataset (cosmopedia-v2 is 28B tokens, far too big to load fully)
ds = load_dataset(
    "HuggingFaceTB/smollm-corpus",
    name="cosmopedia-v2",
    split="train",
    streaming=True,
)
if args.shuffle_buffer > 0:
    ds = ds.shuffle(seed=42, buffer_size=args.shuffle_buffer)

shard_docs = []
shard_chars = 0
shard_idx = 0
total_docs = 0
t0 = time.time()

for doc in ds:
    text = doc["text"]
    shard_docs.append(text)
    shard_chars += len(text)

    enough_chars = shard_chars >= args.chars_per_shard
    aligned = len(shard_docs) % args.row_group_size == 0
    if enough_chars and aligned:
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")
        table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(
            table,
            shard_path,
            row_group_size=args.row_group_size,
            use_dictionary=False,
            compression="zstd",
            compression_level=3,
            write_statistics=False,
        )
        dt = time.time() - t0
        total_docs += len(shard_docs)
        print(f"[{shard_idx + 1}/{args.num_shards}] {shard_path} | docs={len(shard_docs)} | chars={shard_chars:,} | dt={dt:.1f}s | total_docs={total_docs:,}")
        shard_docs = []
        shard_chars = 0
        shard_idx += 1
        t0 = time.time()
        if shard_idx >= args.num_shards:
            break

print(f"Done. Wrote {shard_idx} shards to {output_dir}")
