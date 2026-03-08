from problem import (
    Tree,
    Input,
    reference_kernel,
    Machine
)
from perf_takehome import (
    KernelBuilder,
    DebugInfo
)

import random
random.seed(42)
import copy

# forest_height=10
# rounds=16
# batch_size=256

def run(forest_height, rounds, batch_size):
    forest = Tree.generate(forest_height)
    orig_inp = Input.generate(forest, batch_size, rounds)
    print(f"{forest=}")
    print(f"{orig_inp=}")
    for i in range(1, rounds + 1):
        inp = copy.deepcopy(orig_inp)
        inp.rounds = i
        reference_kernel(forest, inp)
        print(f"round {i}: {inp}")


# print(forest)
# print(inp)

# forest_height=2
# rounds=2
# batch_size=2

# forest=Tree(height=2, values=[3, 0, 8, 7, 7, 4, 3])
# inp=Input(indices=[0, 0, 0, 0], values=[2, 13, 1, 0], rounds=2)
# run(forest_height=2, rounds=2, batch_size=1)
# output: Input(indices=[3, 4, 3, 4], values=[4203971800, 1184307827, 1739325048, 2129651537], rounds=2)



forest_height=2
n_nodes = 2**3
rounds=1
batch_size=1

forest = Tree.generate(forest_height)
orig_inp = Input.generate(forest, batch_size, rounds)

builder = KernelBuilder()
builder.build_kernel(forest_height, len(forest.values), batch_size, rounds)

import json

# print(json.dumps(builder.instrs, indent=2))

# for inst in builder.instrs:
#     print(inst)

# 2KB of memory
mem = [0 for _ in range(16)]

mem[15] = 42

instr = [
    {'store': [('store', 15, 0)]}
    # {'load': [('load', 0, 15)]}
]

machine = Machine(
    mem,
    instr,
    DebugInfo(scratch_map={}),
    n_cores=1,
    value_trace={},
    trace=False,
)

machine.run()

print(mem)