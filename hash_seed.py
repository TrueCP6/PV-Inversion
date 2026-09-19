import os
import sys

def use_same_hash(): # Prevent cache miss warnings - use same hash seed for every rank
    # Kept free of Firedrake/MPI imports: initialising MPI here would leave the
    # parent a singleton MPI process, and any mpiexec it later spawns then fails.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)
