from problem import (
    Tree,
    Input,
    reference_kernel
)

forest_height=10
rounds=16
batch_size=256

forest = Tree.generate(forest_height)
inp = Input.generate(forest, batch_size, rounds)

# print(forest)
# print(inp)

print(reference_kernel(forest, inp))

print(inp)