from cs336_basics import bpe


if __name__ == "__main__":
    # Train BPE on OpenWebText dataset
    input_file = "data/owt_train.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 32000
    merges_filename = "bpe_merges.txt"
    vocab_filename = "bpe_vocab.json"
    out_dir = "models/owt_train_tokenizer"
    vocab, merges = bpe.train_bpe(input_file, vocab_size, special_tokens, 64, 64)
    bpe.save_tokenizer(vocab, merges, out_dir, vocab_filename=vocab_filename, merges_filename=merges_filename)

"""
        Command being timed: "uv run train_bpe_expts_owt.py"
        User time (seconds): 10511.21
        System time (seconds): 2327.38
        Percent of CPU this job got: 190%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 1:52:27
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 21251760
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 0
        Minor (reclaiming a frame) page faults: 491735824
        Voluntary context switches: 143207893
        Involuntary context switches: 109332
        Swaps: 0
        File system inputs: 72
        File system outputs: 1808
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096d
        Exit status: 0
"""
