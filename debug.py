from problem import (
    Tree,
    Input,
    reference_kernel,
    Machine
)
from perf_takehome import (
    KernelBuilder,
    DebugInfo,
    do_kernel_test
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

kb = KernelBuilder()
# builder.build_kernel(forest_height, len(forest.values), batch_size, rounds)

import json

# print(json.dumps(builder.instrs, indent=2))

# for inst in builder.instrs:
#     print(inst)

# 2KB of memory
mem = [0 for _ in range(16)]

mem[12] = 41
mem[13] = 42
mem[14] = 43
mem[15] = 44

instr = [
    # Program #1: Load 15 into scratch, then write it to memory[4]
    # Load 15 into scratch[0]
    # {'load': [('const', 0, 15)]},
    # Load 4 into scratch[1]
    # {'load': [('const', 1, 4)]},
    # Store scratch[0] into address scratch[1]
    # {'store': [('store', 1, 0)]},
    # {'load': [('load', 0, 15)]}


    # Program 2: Load 42 into scratch, then write it to memory[4]
    # {'load': [('const', 0, 15)]},
    # {'load': [('const', 1, 4)]},
    # {'load': [('load', 2, 0)]},
    # {'store': [('store', 1, 2)]}

    # Program 3: Use vload and vstore - store the vector to the start
    # scratch[0]=dest_addr=8
    # {'load': [('const', 0, 8)]},
    # scratch[1]=dest_addr=0
    # {'load': [('const', 1, 0)]},
    # scratch[2:2+8]=mem[8:16]
    # {'load': [('vload', 2, 0)]},
    # mem[0:8]=scratch[2:2+8]
    # {'store': [('vstore', 1, 2)]}

]

# kb.build_kernel2()
kb.build_kernel(forest_height, len(forest.values), batch_size, rounds)

machine = Machine(
    mem,
    kb.instrs,
    DebugInfo(scratch_map={}),
    n_cores=1,
    value_trace={},
    trace=False,
)

machine.run()

print(machine.cores)
print(machine.mem)


# do_kernel_test(
#     forest_height=2,
#     rounds=1,
#     batch_size=1,
#     prints=True
# )