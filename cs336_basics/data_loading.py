import numpy as np
import torch

"""
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """


def get_batch(
    dataset: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # dataset是一个一维的numpy数组，里面存放的是整数形式的token id
    data_len = len(dataset)
    inputs = np.zeros((batch_size, context_length), dtype=np.int64)
    labels = np.zeros((batch_size, context_length), dtype=np.int64)

    for i in range(batch_size):
        start_idx = np.random.randint(0, data_len - context_length)  # 注意左闭右开
        end_idx = start_idx + context_length
        inputs[i] = dataset[start_idx:end_idx]
        labels[i] = dataset[start_idx + 1 : end_idx + 1]

    input_tensor = torch.from_numpy(inputs).long().to(device)
    label_tensor = torch.from_numpy(labels).long().to(device)

    return input_tensor, label_tensor
