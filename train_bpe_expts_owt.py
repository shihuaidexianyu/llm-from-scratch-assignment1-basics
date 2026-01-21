from cs336_basics import bpe


if __name__ == "__main__":
    # Train BPE on OpenWebText dataset
    input_file = "data/owt_train.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 32000
    merges_filename = "bpe_merges.txt"
    vocab_filename = "bpe_vocab.json"
    out_dir = "models/owt_train_tokenizer"
    vocab, merges = bpe.train_bpe(input_file, vocab_size, special_tokens, 4)
    bpe.save_tokenizer(vocab, merges, out_dir, vocab_filename=vocab_filename, merges_filename=merges_filename)
