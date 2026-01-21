from cs336_basics import bpe


if __name__ == "__main__":
    # Train BPE on TinyStories dataset
    input_file = "data/tinystories.txt"
    vocab_size = 10000
    output_codes_file = "models/tinystories_bpe_codes.txt"
    output_vocab_file = "models/tinystories_bpe_vocab.txt"

    bpe.train_bpe(input_file, vocab_size, output_codes_file, output_vocab_file)