
⬜️ Confirm: Can we overlap ops across multiple engines?

⬜️ Parallelize across the batch dimension

⬜️ Understand how flow works

✅ Understand vectorized ops - vload and vstore

VLEN=8 - hardcoded


✅ Understand basic load and store ops

✅ Understand the hashing algorithm

✅ Understand tree traversal

Let's consider height 2, rounds=2 case first


Baseline: 
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
.Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
Speedup over baseline:  1.0