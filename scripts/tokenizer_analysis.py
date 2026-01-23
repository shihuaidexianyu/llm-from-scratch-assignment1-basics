"""Tokenizer analysis script for the assignment.

Performs:
(a) Sample 10 documents from TinyStories and OpenWebText; encode with respective tokenizers and compute bytes/token (compression ratio).
(b) Tokenize OpenWebText with TinyStories tokenizer and compare compression ratio / qualitative note.
(c) Measure tokenizer throughput (bytes/sec) and estimate time to tokenize 825GB.
(d) Encode full training / dev datasets to uint16 NumPy arrays and save to disk.

Usage: run from repository root: python scripts/tokenizer_analysis.py

This script relies on the repository's BPE tokenizer implementation in `cs336_basics/bpe.py`.
"""

import json
import time
from pathlib import Path
# no typing imports needed; use built-in list/tuple annotations
import numpy as np

# Attempt to import repository tokenizer implementation
try:
    from cs336_basics.bpe import BPETokenizer
except Exception:
    BPETokenizer = None

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# Paths for tokenizers
TS_TOKENIZER_DIR = MODELS_DIR / "tinystories_train_tokenizer"
OWT_TOKENIZER_DIR = MODELS_DIR / "owt_train_tokenizer"

# Files to encode
OWT_TRAIN = DATA_DIR / "owt_train.txt"
OWT_VALID = DATA_DIR / "owt_valid.txt"
TS_TRAIN = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
TS_VALID = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"

SAMPLE_DOCS = 10


def load_tokenizer(tokenizer_dir: Path):
    """Load a BPE tokenizer given its directory (expects bpe_vocab.json and bpe_merges.txt).

    Returns a BPETokenizer instance if available, otherwise a minimal fallback that maps
    tokens via the vocab keys (no merges applied).
    """
    vocab_path = tokenizer_dir / "bpe_vocab.json"
    merges_path = tokenizer_dir / "bpe_merges.txt"

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    if BPETokenizer is not None:
        try:
            if merges_path.exists():
                tok = BPETokenizer.from_files(str(vocab_path), str(merges_path))
                return tok
            else:
                # load vocab manually and construct BPETokenizer with empty merges
                from cs336_basics.bpe import RWHelper
                rw = RWHelper()
                vocab_json = json.loads(vocab_path.read_text(encoding="utf-8"))
                vocab = {int(v): rw.decode_token_str(k) for k, v in vocab_json.items()}
                tok = BPETokenizer(vocab, [], [])
                return tok
        except Exception:
            pass

    # Fallback: simple whitespace tokenizer mapping substrings to nearest vocab tokens
    class FallbackTokenizer:
        def __init__(self, vocab):
            # vocab maps token_str -> id
            self.vocab = vocab
            self.id_of = {k: v for k, v in vocab.items()}

    def encode(self, text: str) -> list[int]:
            # naive: split on whitespace and map; unknown words become a sequence of characters if present in vocab
            ids = []
            for word in text.split():
                if word in self.vocab:
                    ids.append(self.vocab[word])
                else:
                    # try characters
                    for ch in word:
                        if ch in self.vocab:
                            ids.append(self.vocab[ch])
                        else:
                            # use 0 if unknown
                            ids.append(0)
            return ids

    return FallbackTokenizer(vocab)


def sample_documents(path: Path, n: int) -> list[str]:
    """Return first n non-empty lines/documents from a file."""
    docs = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                docs.append(s)
                if len(docs) >= n:
                    break
    return docs


def bytes_per_token_for_texts(tokenizer, texts: list[str]) -> tuple[float, int, int]:
    """Compute bytes per token for given texts using tokenizer.encode.

    Returns (bytes_per_token, total_bytes, total_tokens)
    """
    total_bytes = 0
    total_tokens = 0
    for t in texts:
        encoded = tokenizer.encode(t)
        total_tokens += len(encoded)
        total_bytes += len(t.encode("utf-8"))
    bpt = total_bytes / total_tokens if total_tokens > 0 else float('inf')
    return bpt, total_bytes, total_tokens


def measure_throughput(tokenizer, texts: list[str], repeats: int = 3) -> float:
    """Measure bytes/sec tokenization throughput by encoding the concatenation of texts multiple times.

    Returns bytes/sec (average over repeats).
    """
    concat = "\n".join(texts)
    data_bytes = len(concat.encode("utf-8"))
    times = []
    for _ in range(repeats):
        t0 = time.time()
        tokenizer.encode(concat)
        t1 = time.time()
        elapsed = t1 - t0
        if elapsed > 0:
            times.append(data_bytes / elapsed)
    return float(np.mean(times)) if times else 0.0


def save_as_uint16(ids: list[int], out_path: Path):
    arr = np.array(ids, dtype=np.uint16)
    np.save(out_path, arr)


def main():
    print("Loading tokenizers...")
    ts_tok = load_tokenizer(TS_TOKENIZER_DIR)
    owt_tok = load_tokenizer(OWT_TOKENIZER_DIR)

    print("Sampling documents...")
    ts_docs = sample_documents(TS_TRAIN, SAMPLE_DOCS)
    owt_docs = sample_documents(OWT_TRAIN, SAMPLE_DOCS)

    print("(a) Compression ratios (bytes/token)")
    ts_bpt, ts_bytes, ts_toks = bytes_per_token_for_texts(ts_tok, ts_docs)
    owt_bpt, owt_bytes, owt_toks = bytes_per_token_for_texts(owt_tok, owt_docs)
    print(f"TinyStories tokenizer on TinyStories sample: {ts_bpt:.3f} bytes/token (\n{ts_toks} tokens, {ts_bytes} bytes)")
    print(f"OpenWebText tokenizer on OpenWebText sample: {owt_bpt:.3f} bytes/token (\n{owt_toks} tokens, {owt_bytes} bytes)")

    print("(b) Tokenize OpenWebText using TinyStories tokenizer")
    owt_with_ts_bpt, _, _ = bytes_per_token_for_texts(ts_tok, owt_docs)
    print(f"TinyStories tokenizer on OpenWebText sample: {owt_with_ts_bpt:.3f} bytes/token (compare to {owt_bpt:.3f})")

    print("(c) Measure throughput (bytes/sec) and estimate time for 825GB")
    # Use a slightly larger sample for throughput measurement
    sample_for_throughput = sample_documents(OWT_TRAIN, 200)
    ts_throughput = measure_throughput(ts_tok, sample_for_throughput)
    owt_throughput = measure_throughput(owt_tok, sample_for_throughput)
    gb = 825
    seconds_ts = (gb * 1024**3) / ts_throughput if ts_throughput > 0 else float('inf')
    seconds_owt = (gb * 1024**3) / owt_throughput if owt_throughput > 0 else float('inf')
    def secs_to_hours(s):
        return s / 3600.0
    print(f"TinyStories tokenizer throughput: {ts_throughput:.0f} bytes/sec (~{secs_to_hours(seconds_ts):.1f} hours for 825GB)")
    print(f"OpenWebText tokenizer throughput: {owt_throughput:.0f} bytes/sec (~{secs_to_hours(seconds_owt):.1f} hours for 825GB)")

    print("(d) Encode full datasets and save as uint16 numpy arrays (this will process files and may take time)")
    # Example: encode and save small validation files to avoid long runs in normal use
    for txt_path, out_name, tokenizer in [
        (TS_TRAIN, "tinystories_train_uint16.npy", ts_tok),
        (TS_VALID, "tinystories_valid_uint16.npy", ts_tok),
        (OWT_TRAIN, "owt_train_uint16.npy", owt_tok),
        (OWT_VALID, "owt_valid_uint16.npy", owt_tok),
    ]:
        if not txt_path.exists():
            print(f"Skipping {txt_path} (not found)")
            continue
        print(f"Encoding {txt_path} with tokenizer and saving to models/{out_name}...")
        all_ids = []
        with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = tokenizer.encode(line)
                all_ids.extend(ids)
        out_path = MODELS_DIR / out_name
        save_as_uint16(all_ids, out_path)
        print(f"Saved {len(all_ids)} token ids to {out_path} ({out_path.stat().st_size} bytes)")

    print("Done.")


if __name__ == '__main__':
    main()
