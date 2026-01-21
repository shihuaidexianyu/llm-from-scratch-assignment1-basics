from cs336_basics import bpe


if __name__ == "__main__":
    # Train BPE on TinyStories dataset
    input_file = "data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 10000
    merges_filename = "bpe_merges.txt"
    vocab_filename = "bpe_vocab.json"
    out_dir = "models/tinystories_train_tokenizer"
    vocab, merges = bpe.train_bpe(input_file, vocab_size, special_tokens, 16)
    bpe.save_tokenizer(vocab, merges, out_dir, vocab_filename=vocab_filename, merges_filename=merges_filename)

"""
❯ /usr/bin/time -v uv run train_bpe_tinystories.py
        Command being timed: "uv run train_bpe_tinystories.py"
        User time (seconds): 1300.45
        System time (seconds): 14.72
        Percent of CPU this job got: 203%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 10:45.57
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 749436
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 0
        Minor (reclaiming a frame) page faults: 2753015
        Voluntary context switches: 1049
        Involuntary context switches: 22796
        Swaps: 0
        File system inputs: 936952
        File system outputs: 536
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0

the longest tokens are: 

Ġaccomplishment
Ġdisappointment
Ġresponsibility

their lengths area all 15.
"""
