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


def _split_text_with_special_tokens(text: str, special_tokens: list[str]) -> list[tuple[str, bool]]:
    """
    切分文本为 (segment, is_special) 列表，同时保留特殊 token 信息。
    """
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
        # 对于非特殊段做正则切分
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


class RWHelper:
    def __init__(self):
        def _bytes_to_unicode() -> dict[int, str]:
            """
            把 0–255 的每个字节映射成一个“可打印、可存到 JSON 的 Unicode 字符
            参考 GPT-2 的实现
            这样做是为了确保分词后的 token 可以直接作为字符串存储和处理
            例如，字节 0 映射为 Unicode 字符 '\u0100'，字节 255 映射为 '\u01ff'。
            这样就避免了控制字符和不可打印字符的问题。
            该映射是双向的，可以通过反向映射将 Unicode 字符转换回原始字节。
            该函数返回一个字典，键是字节，值是对应的 Unicode 字符。
            """
            bs = (
                list(range(ord("!"), ord("~") + 1))
                + list(range(ord("¡"), ord("¬") + 1))
                + list(range(ord("®"), ord("ÿ") + 1))
            )
            cs = bs[:]
            n = 0
            for b in range(2**8):
                if b not in bs:
                    bs.append(b)
                    cs.append(2**8 + n)
                    n += 1
            return dict(zip(bs, [chr(n) for n in cs]))

        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    def encode_token_bytes(self, token_bytes: bytes) -> str:
        """"""
        return "".join(self.byte_encoder[b] for b in token_bytes)

    def decode_token_str(self, token_str: str) -> bytes:
        return bytes(self.byte_decoder[ch] for ch in token_str)


def save_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    out_dir: str | os.PathLike | Path,
    vocab_filename: str = "tokenizer_vocab.json",
    merges_filename: str = "tokenizer_merges.txt",
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rw_helper = RWHelper()
    vocab_out = {rw_helper.encode_token_bytes(token_bytes): token_id for token_id, token_bytes in vocab.items()}
    vocab_path = out_dir / vocab_filename
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab_out, f, ensure_ascii=False, indent=2)

    merges_path = out_dir / merges_filename
    with merges_path.open("w", encoding="utf-8") as f:
        for token_a, token_b in merges:
            token_a_str = rw_helper.encode_token_bytes(token_a)
            token_b_str = rw_helper.encode_token_bytes(token_b)
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
        rw_helper = RWHelper()
        # 词表是json文件
        # 存放的是 str -> int 映射
        with open(vocab_filepath, encoding="utf-8") as vf:
            vocab_json = json.load(vf)
            for token_str, token_id in vocab_json.items():
                token_id = int(token_id)
                token_bytes = rw_helper.decode_token_str(token_str)
                vocab[token_id] = token_bytes
        merges = []
        # merges 是txt文件
        # 存放是 bytes 对形式
        with open(merges_filepath, encoding="utf-8") as mf:
            for line in mf:
                token1, token2 = line.strip().split()
                merges.append((rw_helper.decode_token_str(token1), rw_helper.decode_token_str(token2)))
        return cls(vocab, merges, special_tokens)

    def save(
        self,
        out_dir: str | os.PathLike | Path,
        vocab_filename: str = "tokenizer_vocab.json",
        merges_filename: str = "tokenizer_merges.txt",
    ) -> tuple[Path, Path]:
        return save_tokenizer(self.vocab, self.merges, out_dir, vocab_filename, merges_filename)

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
            # 将每个字节映射为对应的 token ID
            # 例如 b"Hello" -> (72, 101, 108, 108, 111) -> (id(72), id(101), id(108), id(108), id(111))
            mapped_tokens.append(tuple(self.token_to_id[bytes([b])] for b in token_bytes))
        bpe_tokens = self._apply_merges(mapped_tokens)
        # 展平为单一 ID 列表
        flat_ids = []
        for token in bpe_tokens:
            flat_ids.extend(token)
        return flat_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # 之前没用过 yield from 语法，补充注释说明
        # 这里是“生成器函数”，逐条处理输入文本，逐个产出 token id
        # iterable 可以是：列表、文件对象(逐行)、任何可迭代的字符串序列
        # yield from 等价于：把 self.encode(text) 里的每个 id 逐个 yield 出去
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


class _ReverseBytes:
    __slots__ = ("value",)

    def __init__(self, value: bytes):
        self.value = value

    def __lt__(self, other: "_ReverseBytes") -> bool:
        return self.value > other.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _ReverseBytes):
            return False
        return self.value == other.value


class BPETrainer:
    def __init__(
        self,
        vocab_counts: dict[tuple[int, ...], int],
        special_tokens_ids: set[int],
        merge_workers: int = 1,
    ):
        """
        vocab_counts: 预分词后的词频字典
        special_tokens_ids: 不参与合并的特殊 token 集合
        """
        # 将数据转为可变的 list 结构以便原地修改
        # self.words: list[list[int]]，存储每个词的 token 序列
        # self.freqs: list[int]，存储每个词的频率
        self.words = []
        self.freqs = []
        self.special_tokens_ids = special_tokens_ids
        self.merge_workers = max(1, int(merge_workers))

        for tokens, freq in vocab_counts.items():
            if not tokens:
                continue
            self.words.append(list(tokens))
            self.freqs.append(freq)

        # 倒排索引: token_id -> set(word_index)
        # 记录某个 token 出现在了哪些单词(的索引)中
        self.token_to_word_idxs = collections.defaultdict(set)

        # Pair 统计: (p0, p1) -> count
        self.stats = collections.defaultdict(int)
        # Pair -> set(word_index)
        self.pair_to_word_idxs = collections.defaultdict(set)

        # 堆结构，用于快速取出最高频 pair (lazy deletion)
        # heap item: (-count, token_bytes_a, token_bytes_b, pair)
        self.heap = []

        # 初始化索引和统计
        self._build_index_and_stats()

    def _build_index_and_stats(self):
        for idx, word in enumerate(self.words):
            # 建立倒排索引
            for token in word:
                self.token_to_word_idxs[token].add(idx)
            # 统计初始 Pair
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair[0] in self.special_tokens_ids or pair[1] in self.special_tokens_ids:
                    continue
                self.stats[pair] += self.freqs[idx]
                self.pair_to_word_idxs[pair].add(idx)

    def _iter_word_pairs(self, word: list[int]) -> list[tuple[int, int]]:
        if len(word) < 2:
            return []
        pairs = []
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            if pair[0] in self.special_tokens_ids or pair[1] in self.special_tokens_ids:
                continue
            pairs.append(pair)
        return pairs

    def build_heap(self, id_to_bytes: dict[int, bytes]) -> None:
        import heapq

        self.heap.clear()
        for pair, count in self.stats.items():
            if count <= 0:
                continue
            heapq.heappush(
                self.heap,
                (
                    -count,
                    _ReverseBytes(id_to_bytes.get(pair[0], b"")),
                    _ReverseBytes(id_to_bytes.get(pair[1], b"")),
                    pair,
                ),
            )

    def _remove_word_pairs(self, word: list[int], freq: int) -> None:
        for pair in self._iter_word_pairs(word):
            self.stats[pair] -= freq
            if self.stats[pair] == 0:
                del self.stats[pair]

    def _add_word_pairs(self, word: list[int], freq: int) -> None:
        for pair in self._iter_word_pairs(word):
            self.stats[pair] += freq

    def _get_best_pair(self, id_to_bytes):
        """
        找到频率最高的 Pair。
        Tie-breaking: 频率高优先 -> 字节序大优先
        """
        import heapq

        if not self.stats:
            return None

        while self.heap:
            neg_count, rev_b0, rev_b1, pair = self.heap[0]
            count = self.stats.get(pair, 0)
            if count > 0 and -neg_count == count:
                if rev_b0.value == id_to_bytes.get(pair[0], b"") and rev_b1.value == id_to_bytes.get(pair[1], b""):
                    return pair
            heapq.heappop(self.heap)

        return None

    def merge_pair(self, pair, new_token_id, id_to_bytes: dict[int, bytes]):
        p0, p1 = pair
        if p0 in self.special_tokens_ids or p1 in self.special_tokens_ids:
            return
        import heapq
        from concurrent.futures import ThreadPoolExecutor

        # 优化：只检查真实包含 (p0, p1) 相邻的单词
        candidate_idxs = list(self.pair_to_word_idxs.get(pair, set()))

        def _merge_one(idx: int):
            word = self.words[idx]
            if len(word) < 2:
                return None
            i = 0
            new_word = []
            changed = False
            while i < len(word):
                if i < len(word) - 1 and word[i] == p0 and word[i + 1] == p1:
                    new_word.append(new_token_id)
                    changed = True
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            if not changed:
                return None
            return idx, new_word

        changes = []
        if self.merge_workers > 1 and len(candidate_idxs) > 1:
            with ThreadPoolExecutor(max_workers=self.merge_workers) as executor:
                for result in executor.map(_merge_one, candidate_idxs):
                    if result is not None:
                        changes.append(result)
        else:
            for idx in candidate_idxs:
                result = _merge_one(idx)
                if result is not None:
                    changes.append(result)

        # 执行真正的更新
        updated_pairs = set()

        for idx, new_word in changes:
            freq = self.freqs[idx]
            old_word = self.words[idx]

            # 1. 减去旧统计 (Optimized: 仅减去受影响的部分会更复杂，全量减局部单词其实很快)
            old_pairs = self._iter_word_pairs(old_word)
            for old_pair in old_pairs:
                self.stats[old_pair] -= freq
                if self.stats[old_pair] == 0:
                    del self.stats[old_pair]
                updated_pairs.add(old_pair)
                self.pair_to_word_idxs[old_pair].discard(idx)

            # 2. 更新单词
            self.words[idx] = new_word

            # 3. 加上新统计
            new_pairs = self._iter_word_pairs(new_word)
            for new_pair in new_pairs:
                self.stats[new_pair] += freq
                updated_pairs.add(new_pair)
                self.pair_to_word_idxs[new_pair].add(idx)

            # 4. 更新倒排索引
            old_tokens = set(old_word)
            new_tokens = set(new_word)

            for token in old_tokens - new_tokens:
                self.token_to_word_idxs[token].discard(idx)
            for token in new_tokens - old_tokens:
                self.token_to_word_idxs[token].add(idx)

            self.token_to_word_idxs[new_token_id].add(idx)

        for pair in updated_pairs:
            count = self.stats.get(pair, 0)
            if count > 0:
                heapq.heappush(
                    self.heap,
                    (
                        -count,
                        _ReverseBytes(id_to_bytes.get(pair[0], b"")),
                        _ReverseBytes(id_to_bytes.get(pair[1], b"")),
                        pair,
                    ),
                )


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    num_workers: int = 4,
    merge_workers: int = 1,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = special_tokens or []
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b""

    # 1. Pre-tokenization (保留原有的多进程逻辑，这部分效率已经不错)
    boundaries = find_chunk_boundaries(input_path, num_workers, split_token)
    chunk_args = []
    for i in range(len(boundaries) - 1):
        chunk_args.append((input_path, boundaries[i], boundaries[i + 1], special_tokens))

    global_vocab_counts = collections.Counter()
    if num_workers > 1 and len(chunk_args) > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            for local_counts in pool.imap_unordered(_process_chunk_worker, chunk_args):
                global_vocab_counts.update(local_counts)
    else:
        for args in chunk_args:
            global_vocab_counts.update(_process_chunk_worker(args))

    # 2. 初始化 BPE Trainer
    special_tokens_ids = {256 + i for i in range(len(special_tokens))}
    # 原代码中 _process_chunk_worker 已经将特殊 token 映射为 256+ID

    trainer = BPETrainer(global_vocab_counts, special_tokens_ids, merge_workers=merge_workers)

    # 初始化词表
    vocab = {int(i): bytes([i]) for i in range(0, 256)}
    merges = []

    # 准备 id_to_bytes 映射，用于 tie-breaking 和输出
    id_to_bytes = {i: bytes([i]) for i in range(256)}

    # 特殊 token 处理
    current_token_id = 256
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        vocab[current_token_id] = token_bytes
        id_to_bytes[current_token_id] = token_bytes
        current_token_id += 1

    # 初始化 heap
    trainer.build_heap(id_to_bytes)

    base_vocab_size = current_token_id
    target_merges = vocab_size - base_vocab_size

    if target_merges < 0:
        raise ValueError("vocab_size too small")

    print(f"Starting BPE training. Target merges: {target_merges}")

    # 3. BPE 循环
    for i in range(target_merges):
        if i % 100 == 0:
            print(f"Merge {i}/{target_merges}...")

        # 获取最佳 Pair
        best_pair = trainer._get_best_pair(id_to_bytes)
        if best_pair is None:
            print(f"No more pairs to merge at iteration {i}. Stopping early.")
            break

        p0, p1 = best_pair

        # 记录 Merge 规则
        merges.append((id_to_bytes[p0], id_to_bytes[p1]))

        # 更新 id_to_bytes
        new_token_bytes = id_to_bytes[p0] + id_to_bytes[p1]
        id_to_bytes[current_token_id] = new_token_bytes
        vocab[current_token_id] = new_token_bytes

        # 执行合并
        trainer.merge_pair(best_pair, current_token_id, id_to_bytes)

        current_token_id += 1

    return vocab, merges
