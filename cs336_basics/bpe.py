import os
import json
import multiprocessing
import collections
from pathlib import Path
from typing import BinaryIO
from collections.abc import Iterable, Iterator


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    # pretokenization_example.py 中的函数, 直接复制过来
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def find_chunk_boundaries(
    filename: str,
    desired_num_chunks: int,
    split_special_token: bytes = b"<|endoftext|>",
) -> list[int]:
    """
    计算文件分块边界，尽量对齐到特殊 token，避免切断 token。
    若找不到特殊 token，则退化为单 chunk。
    """
    file_size = os.path.getsize(filename)
    if file_size == 0:
        return [0]
    if not split_special_token:
        return [0, file_size]
    with open(filename, "rb") as f:
        return _find_chunk_boundaries(f, desired_num_chunks, split_special_token)


# 尝试导入 regex 库 (支持 \p{L} 等高级特性)，如果不存在则回退到 re
try:
    import regex as re

    # GPT-2 pattern provided in assignment
    GPT2_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
except ImportError:
    import re

    print("Warning: 'regex' module not found, using standard 're'. Pre-tokenization might differ.")
    GPT2_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")


def _process_chunk_worker(args) -> collections.Counter:
    """
    Worker function: 读取文件的一个片段，正则分词，返回词频统计。
    """
    filename, start, end, special_tokens = args
    local_counts = collections.Counter()
    special_token_ids = {token: 256 + i for i, token in enumerate(special_tokens or [])}

    # "Pre-tokenization" 步骤
    try:
        with open(filename, "rb") as f:
            f.seek(start)
            # 读取字节并 decode
            # errors='ignore' 防止边界处切断了多字节字符导致报错
            text = f.read(end - start).decode("utf-8", errors="ignore")
            for segment, is_special in _split_text_with_special_tokens(text, special_tokens):
                if not segment:
                    continue
                if is_special:
                    token_id = special_token_ids.get(segment)
                    if token_id is not None:
                        local_counts[(token_id,)] += 1
                    else:
                        local_counts[tuple(segment.encode("utf-8"))] += 1
                    continue
                # 正则切分,注意这一步在保留特殊 token 之后进行,否则会把特殊 token 切碎
                tokens = re.findall(GPT2_PAT, segment)
                # 转换为字节元组并统计
                # 例如: "Hello" -> b"Hello" -> (72, 101, 108, 108, 111)
                for token in tokens:
                    token_bytes = tuple(token.encode("utf-8"))
                    local_counts[token_bytes] += 1

    except Exception as e:
        print(f"Error processing chunk {start}-{end}: {e}")

    return local_counts


def _iter_non_special_segments(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text] if text else []
    tokens = sorted(set(special_tokens), key=len, reverse=True)
    # 过滤空 token，避免问题
    tokens = [t for t in tokens if t]
    if not tokens:
        return [text] if text else []
    pattern = "|".join(re.escape(t) for t in tokens)
    regex = re.compile(pattern)
    segments = []
    last_end = 0
    for match in regex.finditer(text):
        if match.start() > last_end:
            segments.append(text[last_end : match.start()])
        last_end = match.end()
    if last_end < len(text):
        segments.append(text[last_end:])
    return segments


def _split_text_with_special_tokens(text: str, special_tokens: list[str]) -> list[tuple[str, bool]]:
    """Split text into (segment, is_special) while preserving special tokens."""
    if not special_tokens:
        return [(text, False)] if text else []
    tokens = sorted(set(special_tokens), key=len, reverse=True)
    tokens = [t for t in tokens if t]
    if not tokens:
        return [(text, False)] if text else []
    pattern = "|".join(re.escape(t) for t in tokens)
    regex = re.compile(pattern)
    segments: list[tuple[str, bool]] = []
    last_end = 0
    for match in regex.finditer(text):
        if match.start() > last_end:
            segments.append((text[last_end : match.start()], False))
        segments.append((match.group(0), True))
        last_end = match.end()
    if last_end < len(text):
        segments.append((text[last_end:], False))
    return segments


def pretokenize_text(text: str, special_tokens: list[str] | None = None) -> list[tuple[int, ...]]:
    """
    对输入文本进行预分词，返回字节 token 列表。
    """
    special_tokens = special_tokens or []
    segments = _split_text_with_special_tokens(text, special_tokens)
    tokens: list[tuple[int, ...]] = []
    for segment, is_special in segments:
        if not segment:
            continue
        if is_special:
            tokens.append(tuple(segment.encode("utf-8")))
            continue
        # 正则切分
        subtokens = re.findall(GPT2_PAT, segment)
        for token in subtokens:
            tokens.append(tuple(token.encode("utf-8")))
    return tokens


def get_stats(vocab_counts: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    """
    统计当前所有词汇中的 pair 频率
    """
    pairs = collections.defaultdict(int)
    for ids, freq in vocab_counts.items():
        for i in range(len(ids) - 1):
            pairs[ids[i], ids[i + 1]] += freq
    return pairs


def merge_vocab(
    best_pair: tuple[int, int], vocab_counts: dict[tuple[int, ...], int], new_token_id: int
) -> dict[tuple[int, ...], int]:
    """
    将词表中所有的 best_pair 替换为 new_token_id
    """
    new_vocab = {}
    p0, p1 = best_pair

    for ids, freq in vocab_counts.items():
        # 优化：如果词里没有 p0，肯定不需要合并，直接跳过计算
        if p0 not in ids:
            new_vocab[ids] = freq
            continue

        new_ids = []
        i = 0
        while i < len(ids):
            # 检查是否匹配 p0, p1
            if i < len(ids) - 1 and ids[i] == p0 and ids[i + 1] == p1:
                new_ids.append(new_token_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        new_vocab[tuple(new_ids)] = freq

    return new_vocab


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    num_workers: int = 4,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # 计算分块边界
    # 假设特殊 token 是 <|endoftext|>，如果不包含，可以传空 bytes 或其他
    special_tokens = special_tokens or []
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b""
    boundaries = find_chunk_boundaries(input_path, num_workers, split_token)
    # 准备参数 [(file, start, end), (file, start, end), ...]
    chunk_args = []
    for i in range(len(boundaries) - 1):
        chunk_args.append((input_path, boundaries[i], boundaries[i + 1], special_tokens))
    # 并行 Pre-tokenization (Map 阶段)
    # 使用 multiprocessing.Pool 自动管理进程
    global_vocab_counts = collections.Counter()
    if num_workers > 1 and len(chunk_args) > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            # imap_unordered 稍微快一点，因为我们不关心顺序
            for local_counts in pool.imap_unordered(_process_chunk_worker, chunk_args):
                global_vocab_counts.update(local_counts)
    else:
        for args in chunk_args:
            global_vocab_counts.update(_process_chunk_worker(args))
    # BPE 迭代 (Serial Merge 阶段)
    # 初始 token 0-255
    # 初始化 vocab，包含基础字节和特殊 token
    vocab = {int(i): bytes([i]) for i in range(0, 256)}
    merges = []
    # 从 base_vocab_size 开始分配新 ID (避免覆盖特殊 token)
    current_token_id = 256 + len(special_tokens)
    # 加上特殊 token 的数量
    base_vocab_size = 256 + len(special_tokens)
    # 计算需要的合并次数
    target_merges = vocab_size - base_vocab_size
    if target_merges < 0:
        raise ValueError("vocab_size too small for base vocabulary + special tokens")
    # 将 Counter 转为普通 dict 以便处理，虽然 Counter 也能用但 dict 更轻量
    vocab_counts = dict(global_vocab_counts)
    id_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    for i, token in enumerate(special_tokens, start=256):
        id_to_bytes[i] = token.encode("utf-8")
        vocab[i] = token.encode("utf-8")
    for i in range(target_merges):
        # a. 统计 Pair 频率
        pairs = get_stats(vocab_counts)
        if not pairs:
            break
        # b. 找到频率最高的 Pair
        # 按照 (频率, 字节序) 排序，保证确定性
        best_pair = max(
            pairs,
            key=lambda p: (
                pairs[p],
                id_to_bytes[p[0]],
                id_to_bytes[p[1]],
            ),
        )
        # c. 记录合并规则（以 bytes 对形式返回）
        merges.append((id_to_bytes[best_pair[0]], id_to_bytes[best_pair[1]]))
        id_to_bytes[current_token_id] = id_to_bytes[best_pair[0]] + id_to_bytes[best_pair[1]]

        # d. 更新词表
        # 这一步是单线程的，但因为是在 len(vocab) 上操作，通常很快
        vocab_counts = merge_vocab(best_pair, vocab_counts, current_token_id)
        vocab[current_token_id] = id_to_bytes[current_token_id]

        current_token_id += 1
    return vocab, merges


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


def _encode_token_bytes(token_bytes: bytes, byte_encoder: dict[int, str]) -> str:
    return "".join(byte_encoder[b] for b in token_bytes)


def save_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    out_dir: Path,
    vocab_filename: str = "tokenizer_vocab.json",
    merges_filename: str = "tokenizer_merges.txt",
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    byte_encoder = _bytes_to_unicode()

    vocab_out = {_encode_token_bytes(token_bytes, byte_encoder): token_id for token_id, token_bytes in vocab.items()}
    vocab_path = out_dir / vocab_filename
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab_out, f, ensure_ascii=False, indent=2)

    merges_path = out_dir / merges_filename
    with merges_path.open("w", encoding="utf-8") as f:
        for token_a, token_b in merges:
            token_a_str = _encode_token_bytes(token_a, byte_encoder)
            token_b_str = _encode_token_bytes(token_b, byte_encoder)
            f.write(f"{token_a_str} {token_b_str}\n")

    return vocab_path, merges_path


class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        # str--->bytes--->int
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        # 构建反向词表和合并规则映射
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.bpe_ranks = {(self.token_to_id[a], self.token_to_id[b]): i for i, (a, b) in enumerate(merges)}
        self.special_tokens_ids = set(self.token_to_id[token.encode("utf-8")] for token in self.special_tokens)
        self.special_tokens_bytes = {tuple(token.encode("utf-8")) for token in self.special_tokens}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = {}  # dict[int, bytes]
        special_tokens = special_tokens or []
        byte_decoder = {v: k for k, v in _bytes_to_unicode().items()}

        def _decode_token_str(token_str: str) -> bytes:
            return bytes(byte_decoder[ch] for ch in token_str)

        # 词表是json文件
        # 存放的是 str -> int 映射
        with open(vocab_filepath, encoding="utf-8") as vf:
            vocab_json = json.load(vf)
            for token_str, token_id in vocab_json.items():
                token_id = int(token_id)
                token_bytes = _decode_token_str(token_str)
                vocab[token_id] = token_bytes
        merges = []
        # merges 是txt文件
        # 存放是 bytes 对形式
        with open(merges_filepath, encoding="utf-8") as mf:
            for line in mf:
                token1, token2 = line.strip().split()
                merges.append((_decode_token_str(token1), _decode_token_str(token2)))
        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _get_pairs(token_ids: tuple[int, ...]) -> set[tuple[int, int]]:
        pairs = set()
        if len(token_ids) < 2:
            return pairs
        prev_id = token_ids[0]
        for curr_id in token_ids[1:]:
            pairs.add((prev_id, curr_id))
            prev_id = curr_id
        return pairs

    def _apply_merges(self, token_ids: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
        merged_tokens: list[tuple[int, ...]] = []
        for token in token_ids:
            # 如果是特殊 token，直接跳过
            if len(token) == 1 and token[0] in self.special_tokens_ids:
                merged_tokens.append(token)
                continue
            pairs = self._get_pairs(token)
            while True:
                # 找到当前 token 中排名最高的 pair
                best_pair = min(
                    (pair for pair in pairs if pair in self.bpe_ranks),
                    key=lambda p: self.bpe_ranks[p],
                    default=None,
                )
                if best_pair is None:
                    break
                # 合并 best_pair
                new_token = []
                i = 0
                p0, p1 = best_pair
                while i < len(token):
                    if i < len(token) - 1 and token[i] == p0 and token[i + 1] == p1:
                        new_token.append(self.token_to_id[self.vocab[p0] + self.vocab[p1]])  # 合并为新token
                        i += 2
                    else:
                        new_token.append(token[i])
                        i += 1
                token = tuple(new_token)
                pairs = self._get_pairs(token)
            merged_tokens.append(token)
        return merged_tokens

    def encode(self, text: str) -> list[int]:
        pretokens = pretokenize_text(text, self.special_tokens)
        mapped_tokens: list[tuple[int, ...]] = []
        for token_bytes in pretokens:
            if not token_bytes:
                continue
            if token_bytes in self.special_tokens_bytes:
                token_id = self.token_to_id[bytes(token_bytes)]
                mapped_tokens.append((token_id,))
                continue
            mapped_tokens.append(tuple(self.token_to_id[bytes([b])] for b in token_bytes))
        bpe_tokens = self._apply_merges(mapped_tokens)
        # 展平为单一 ID 列表
        flat_ids = []
        for token in bpe_tokens:
            flat_ids.extend(token)
        return flat_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        bytes_chunks = []
        for token_id in ids:
            token_bytes = self.vocab.get(token_id, b"")
            bytes_chunks.append(token_bytes)
        text_bytes = b"".join(bytes_chunks)
        # 解码 bytes 为 str
        text = text_bytes.decode("utf-8", errors="ignore")
        return text
