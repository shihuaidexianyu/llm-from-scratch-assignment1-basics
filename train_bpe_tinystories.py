from cs336_basics import bpe


if __name__ == "__main__":
    # Train BPE on TinyStories dataset
    input_file = "data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 10000
    merges_filename = "bpe_merges.txt"
    vocab_filename = "bpe_vocab.json"
    out_dir = "models/tinystories_train_tokenizer"
    vocab, merges = bpe.train_bpe(input_file, vocab_size, special_tokens, 16, 16)
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



系统“内耗”严重 (System Time 爆炸)
程序花了 1889秒 在系统内核上（搬运数据、调度进程），却只花了 846秒 在真正的 BPE 计算上。
这意味着 2/3 的时间 被浪费在了进程间通信（IPC）和内存复制上，而不是在做分词训练。
主进程在“空转”
cProfile 显示，耗时最长的函数全是 multiprocessing 库内部的 wait（等待）、recv（接收数据）和 terminate（关闭进程）。
主进程有 600多秒 只是在等着子进程把处理完的数据传回来，而不是在工作。
"""


"""
❯ cd /home/hw/learn/llm-from-scratch-assignment1-basics && /usr/bin/time -v /home/hw/learn/llm-from-scratch-assignment1-basics/.venv/bin/python -c "
cmdand dquote> import cProfile
cmdand dquote> import pstats
cmdand dquote> from io import StringIO
cmdand dquote> from cs336_basics import bpe
cmdand dquote>
cmdand dquote> # Profile the training
cmdand dquote> pr = cProfile.Profile()
cmdand dquote> pr.enable()
cmdand dquote> vocab, merges = bpe.train_bpe('data/TinyStoriesV2-GPT4-train.txt', 1000, ['<|endoftext|>'], 4)
cmdand dquote> pr.disable()
cmdand dquote>
cmdand dquote> # Get stats
cmdand dquote> s = StringIO()
cmdand dquote> ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
cmdand dquote> ps.print_stats(20) # Top 20 functions by cumulative time
cmdand dquote> print(s.getvalue())
cmdand dquote> "
    104291092 function calls (104290978 primitive calls) in 934.797 seconds

 Ordered by: cumulative time
 List reduced from 415 to 20 due to restriction <20>

 ncalls tottime percall cumtime percall filename:lineno(function)
    1  1.004  1.004 934.797 934.797 /home/hw/learn/llm-from-scratch-assignment1-basics/cs336_basics/bpe.py:213(train_bpe)
    1  0.000  0.000 601.369 601.369 /home/hw/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:738(__exit__)
    1  0.000  0.000 601.356 601.356 /home/hw/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:654(terminate)
    5  0.000  0.000 601.348 120.270 /home/hw/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/multiprocessing/util.py:272(__call__)
    1  0.000  0.000 601.348 601.348 /home/hw/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:680(_terminate_pool)
    5  0.000  0.000 601.293 120.259 /home/hw/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/multiprocessing/connection.py:246(recv)
  58/54  0.003  0.000 601.252 11.134 {built-in method posix.read}
    1  0.000  0.000 601.249 601.249 /home/hw/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:671(_help_stuff_finish)
    1  0.000  0.000 601.249 601.249 {method 'acquire' of '_multiprocessing.SemLock' objects}
   3/1  0.000  0.000 601.248 601.248 /home/hw/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/threading.py:1000(_bootstrap)
   3/1  0.001  0.000 601.248 601.248 /home/hw/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/threading.py:1027(_bootstrap_inner)
   3/1  0.001  0.000 601.248 601.248 /home/hw/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/threading.py:983(run)
    1  0.000  0.000 601.248 601.248 /home/hw/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:573(_handle_results)
  14/10  0.000  0.000 601.247 60.125 /home/hw/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/multiprocessing/connection.py:390(_recv)
   7/5  0.000  0.000 601.247 120.249 /home/hw/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/multiprocessing/connection.py:429(_recv_bytes)
   440  0.002  0.000 601.014  1.366 /home/hw/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:333(_maintain_pool)
   440  0.008  0.000 600.961  1.366 /home/hw/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:289(_join_exited_workers)
   440  0.003  0.000 600.793  1.365 /home/hw/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/multiprocessing/pool.py:500(_wait_for_updates)
   883  0.017  0.000 600.649  0.680 /home/hw/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/multiprocessing/connection.py:1134(wait)
   883  0.005  0.000 600.511  0.680 /home/hw/.local/share/uv/python/cpython-3.13.9-linux-x86_64-gnu/lib/python3.13/selectors.py:385(select)



    Command being timed: "/home/hw/learn/llm-from-scratch-assignment1-basics/.venv/bin/python -c
import cProfile
import pstats
from io import StringIO
from cs336_basics import bpe

# Profile the training
pr = cProfile.Profile()
pr.enable()
vocab, merges = bpe.train_bpe('data/TinyStoriesV2-GPT4-train.txt', 1000, ['<|endoftext|>'], 4)
pr.disable()

# Get stats
s = StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(20) # Top 20 functions by cumulative time
print(s.getvalue())
"
    User time (seconds): 846.46
    System time (seconds): 1889.52
    Percent of CPU this job got: 292%
    Elapsed (wall clock) time (h:mm:ss or m:ss): 15:34.90
    Average shared text size (kbytes): 0
    Average unshared data size (kbytes): 0
    Average stack size (kbytes): 0
    Average total size (kbytes): 0
    Maximum resident set size (kbytes): 3454988
    Average resident set size (kbytes): 0
    Major (requiring I/O) page faults: 7
    Minor (reclaiming a frame) page faults: 3731550
    Voluntary context switches: 5179
    Involuntary context switches: 10327
    Swaps: 0
    File system inputs: 1080936
    File system outputs: 0
    Socket messages sent: 0
    Socket messages received: 0
    Signals delivered: 0
    Page size (bytes): 4096
    Exit status: 0
"""
