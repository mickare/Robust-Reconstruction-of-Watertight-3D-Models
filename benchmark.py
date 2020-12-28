import timeit

setup = """
import numpy as np
x = (np.random.random((20,20,20)) > 0.5).astype(np.int8)
"""

stmt0 = """
i = np.nonzero(x)
"""

stmt1 = """
i = np.argwhere(x)
"""

print("stmt0: ", timeit.timeit(stmt0, setup=setup, number=20000))
print("stmt1: ", timeit.timeit(stmt1, setup=setup, number=20000))
