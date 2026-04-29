"""
Stream mlfoundations/dclm-baseline-1.0 (ML-classifier-filtered web text)
and write parquet shards matching the ClimbMix shard schema.
"""
import os
import time
import argparse

from datasets import load_dataset
import pyarrow.parquet as pq
import pyarrow as pa

parser = argparse.ArgumentParser()
parser.add_argument("--num-shards", type=int, default=24)
parser.add_argument("--chars-per-shard", type=int, default=250_000_000)
parser.add_argument("--row-group-size", type=int, default=1024)
parser.add_argument("--output-dirname", type=str, default="base_data_dclm")
parser.add_argument("--shuffle-buffer", type=int, default=10_000)
args = parser.parse_args()

base_dir = os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))
output_dir = os.path.join(base_dir, args.output_dirname)
os.makedirs(output_dir, exist_ok=True)
print(f"Output: {output_dir}")
print(f"Target: {args.num_shards} shards x {args.chars_per_shard:,} chars")

ds = load_dataset(
    "mlfoundations/dclm-baseline-1.0-parquet",
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
    text = doc.get("text") or doc.get("content")
    if not text:
        continue
    shard_docs.append(text)
    shard_chars += len(text)
    if shard_chars >= args.chars_per_shard and len(shard_docs) % args.row_group_size == 0:
        path = os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(
            pa.Table.from_pydict({"text": shard_docs}),
            path,
            row_group_size=args.row_group_size,
            use_dictionary=False,
            compression="zstd",
            compression_level=3,
            write_statistics=False,
        )
        dt = time.time() - t0
        total_docs += len(shard_docs)
        print(f"[{shard_idx + 1}/{args.num_shards}] {path} | docs={len(shard_docs)} | chars={shard_chars:,} | dt={dt:.1f}s")
        shard_docs = []
        shard_chars = 0
        shard_idx += 1
        t0 = time.time()
        if shard_idx >= args.num_shards:
            break

print(f"Done. Wrote {shard_idx} shards to {output_dir}")
