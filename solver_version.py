"""Fingerprint numerical sources so caches cannot cross solver revisions."""
from pathlib import Path
import hashlib
_root=Path(__file__).resolve().parent
SOLVER_REVISION=hashlib.sha256(b'\n'.join((_root/name).read_bytes() for name in
    ['starmodel.py','fast_kernel.py','fast_solver.py','fast_grid.py','model_runner.py'])).hexdigest()
