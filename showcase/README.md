# Showcase: videos and animations

Scripts that run the real solvers, record what they do, and render the videos and README animations in `docs/media/`.
They are kept apart from the thesis code: capture scripts import the solver modules unchanged, and the renderers only read
the `.npz` files the capture scripts write.

| Script | Where it runs | What it does |
|---|---|---|
| `smoke_test.py` | Firedrake container | Inverts the control state and prints the headline surface diagnostics |
| `capture_diagnostic.py` | Firedrake container | Inverts the control state (N = 40, p = 4) with a KSP monitor that keeps every CG iterate, then solves the same system with Jacobi-preconditioned CG for comparison, and inverts the background alone to isolate the intrusion's own response |
| `capture_prognostic.py` | Firedrake container | Steps the control state for 168 h on `timestepping.py`'s domain at reduced resolution, saving one frame per simulated hour |
| `render_diagnostic.py`, `render_prognostic.py` | Host (no Firedrake) | Render the two videos with matplotlib and PyVista, piped to ffmpeg |
| `make_readme_media.py` | Host | Cuts the README GIFs and stills from the videos, and the thesis figures from `docs/thesis.pdf` |
| `videokit.py`, `sampling.py`, `thesis_figures.py` | | Shared helpers |

## Rebuilding

```bash
# 1. Capture: about 20 min for the diagnostic run, then about 3 h for the 168 h prognostic run
#    (4 cores; the prognostic run in docs/media used 3 ranks so a 4th core was free for rendering)
docker run --rm -v "$PWD":/work -v "$PWD/showcase_data":/data -w /work -e PYTHONPATH=/work \
    firedrakeproject/firedrake-vanilla-default:latest bash -c '
      mpiexec -n 4 python3 showcase/capture_diagnostic.py --out /data &&
      mpiexec -n 4 python3 showcase/capture_prognostic.py --out /data'

# 2. Render (host: Python 3 with numpy, scipy, matplotlib, pyvista, pymupdf; ffmpeg; OSMesa or EGL for headless VTK)
cd showcase
python render_diagnostic.py ../showcase_data ../docs/media/diagnostic_solver.mp4 --stills ../showcase_data/stills
python render_prognostic.py ../showcase_data ../docs/media/prognostic_solver.mp4 --stills ../showcase_data/stills
python make_readme_media.py ../showcase_data ../docs/media
```

The race between the two CG solvers is timed on the wall clock, so run the diagnostic capture on an otherwise idle
machine. More generally, keep other work off the cores the MPI ranks use (`docker run --cpuset-cpus`, `taskset`):
one descheduled rank stalls all the others, and a shared core made each time step about 6× slower here. The renders use the [Inter](https://rsms.me/inter/) font when it is installed (`fonts-inter` on Debian/Ubuntu).
