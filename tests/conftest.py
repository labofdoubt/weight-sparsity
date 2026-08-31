import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Every model in this suite is deliberately tiny (d_model=32, micro batches of 4)
# and runs on CPU, so each op is a few microseconds of work.  Left at the default
# -- one thread per core -- the OpenMP fork/join barrier costs far more than the
# op itself, and on a many-core host the suite spins instead of computing: the
# whole run took 2h25m at 192 threads versus ~12 CPU-seconds of real work in its
# slowest test.  Cap the pools before torch is imported (conftest is loaded ahead
# of the test modules), but leave an explicit caller setting alone.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import torch  # noqa: E402  (must follow the env vars above)

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # already fixed once the interop pool has started; the env vars still apply
