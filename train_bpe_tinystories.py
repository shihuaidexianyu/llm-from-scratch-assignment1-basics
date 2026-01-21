from cs336_basics import bpe


if __name__ == "__main__":
    # Train BPE on TinyStories dataset
    input_file = "data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 10000
    merges_filename = "bpe_merges.txt"
    vocab_filename = "bpe_vocab.txt"
    out_dir = "models/tinystories_train_tokenizer"
    vocab, merges = bpe.train_bpe(input_file, vocab_size, special_tokens, 16)
    bpe.save_tokenizer(vocab, merges, out_dir, vocab_filename=vocab_filename, merges_filename=merges_filename)
