from cs336_basics import bpe

spcial_tokens = ["<|endoftext|>"]
owt_merges = "models/owt_train_tokenizer/bpe_merges.txt"
owt_vocab = "models/owt_train_tokenizer/bpe_vocab.json"
owt_tokenlizer = bpe.BPETokenizer.from_files(owt_vocab, owt_merges, special_tokens=spcial_tokens)
owt_test_sentences = [
    "This is a test sentence.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
]


tinystories_vocab = "models/tinystories_train_tokenizer/bpe_vocab.json"
tinystories_merges = "models/tinystories_train_tokenizer/bpe_merges.txt"
tinystories_tokenlizer = bpe.BPETokenizer.from_files(
    tinystories_vocab, tinystories_merges, special_tokens=spcial_tokens
)
tinystories_test_sentences = [
    "Once upon a time, there was a little girl named Lily.",
    "She lived in a small village surrounded by mountains.",
    "Every day, she would explore the forest and discover new adventures.",
]


def experiment_tokenizers(tokenizer, test_sentences):
    for sentence in test_sentences:
        tokens = tokenizer.tokenize(sentence)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        decoded_sentence = tokenizer.decode(token_ids)
        print(f"Original Sentence: {sentence}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Decoded Sentence: {decoded_sentence}")
        print("-" * 50)


if __name__ == "__main__":
    print("Experimenting with OpenWebText Tokenizer:")
    experiment_tokenizers(owt_tokenlizer, owt_test_sentences)
    print("\nExperimenting with TinyStories Tokenizer:")
    experiment_tokenizers(tinystories_tokenlizer, tinystories_test_sentences)
