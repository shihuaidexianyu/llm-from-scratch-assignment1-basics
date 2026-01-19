import os
import multiprocessing
import collections
from typing import BinaryIO


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

    # "Pre-tokenization" 步骤
    try:
        with open(filename, "rb") as f:
            f.seek(start)
            # 读取字节并 decode
            # errors='ignore' 防止边界处切断了多字节字符导致报错
            text = f.read(end - start).decode("utf-8", errors="ignore")
            for segment in _iter_non_special_segments(text, special_tokens):
                # 正则切分,注意这一步在去除特殊 token 之后进行,否则会把特殊 token 切碎
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
