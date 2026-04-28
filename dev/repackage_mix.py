"""
Build a mixed-source training dataset by interleaving documents from existing
parquet shard sets. Source-fraction is configurable per dir. Output shards
match the ClimbMix shard schema (single 'text' column).

Reads existing pre-packaged dirs from $NANOCHAT_BASE_DIR/<source>/shard_*.parquet
so it has zero HF download cost — assumes both sources have already been
downloaded once via repackage_cosmopedia.py / repackage_fineweb_edu.py /
the ClimbMix downloader.

Convention: --output-dirname is parsed for source pcts, e.g.
  base_data_mix_climb90_cosmo10  -> 90% climbmix + 10% cosmopedia_v2
  base_data_mix_climb80_cosmo20
  base_data_mix_climb50_fwe50
"""
import os
import re
import time
import random
import argparse

import pyarrow.parquet as pq
import pyarrow as pa

SOURCE_ALIAS = {
    "climb": "base_data_climbmix",
    "cosmo": "base_data_cosmopedia_v2",
    "fwe": "base_data_fineweb_edu",
}

parser = argparse.ArgumentParser()
parser.add_argument("--output-dirname", type=str, required=True,
                    help="must encode mix, e.g. base_data_mix_climb90_cosmo10")
parser.add_argument("--num-shards", type=int, default=24)
parser.add_argument("--chars-per-shard", type=int, default=250_000_000)
parser.add_argument("--row-group-size", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Parse mix recipe out of dirname: e.g. base_data_mix_climb90_cosmo10
m = re.fullmatch(r"base_data_mix(?:_([a-z]+)(\d+))(?:_([a-z]+)(\d+))?(?:_([a-z]+)(\d+))?", args.output_dirname)
if not m:
    raise SystemExit(f"Could not parse mix recipe from {args.output_dirname}")
mix = []
for i in range(0, len(m.groups()), 2):
    src, pct = m.group(i + 1), m.group(i + 2)
    if src is None:
        continue
    if src not in SOURCE_ALIAS:
        raise SystemExit(f"Unknown source alias '{src}'. Known: {list(SOURCE_ALIAS)}")
    mix.append((SOURCE_ALIAS[src], int(pct)))
total_pct = sum(p for _, p in mix)
if total_pct != 100:
    raise SystemExit(f"Mix percentages must sum to 100, got {total_pct} from {mix}")
print(f"Mix recipe: {mix}")

base_dir = os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))
output_dir = os.path.join(base_dir, args.output_dirname)
os.makedirs(output_dir, exist_ok=True)

def doc_iter(source_dirname):
    src_dir = os.path.join(base_dir, source_dirname)
    files = sorted([f for f in os.listdir(src_dir) if f.endswith('.parquet')])
    for f in files:
        pf = pq.ParquetFile(os.path.join(src_dir, f))
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column('text').to_pylist():
                yield text

iters = {src: doc_iter(src) for src, _ in mix}
weights = [(src, p) for src, p in mix]
rng = random.Random(args.seed)

shard_docs = []
shard_chars = 0
shard_idx = 0
total_docs = 0
t0 = time.time()
done = False

while not done:
    # Sample a source by weight, pull next doc
    population = [src for src, _ in weights]
    pcts = [p for _, p in weights]
    src = rng.choices(population, weights=pcts, k=1)[0]
    try:
        text = next(iters[src])
    except StopIteration:
        # That source exhausted; rebuild iterator (loop)
        iters[src] = doc_iter(src)
        text = next(iters[src])
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
            done = True
            break

print(f"Done. Wrote {shard_idx} shards to {output_dir}")
