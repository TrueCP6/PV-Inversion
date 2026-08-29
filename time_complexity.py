import argparse
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import sweep

# Define a way to store run results
@dataclass
class TimeRecord:
    initial_run : bool
    p : int
    matfree : bool
    N : int
    time : float

    def dofs(self):
        return sweep.dof_count(self.p, self.N)

def _run_point(args):
    """Time repeated solves at one (N, matfree) point and write them to args.out."""
    sweep.quiet_petsc()

    solver = sweep.build_solver(args.N, args.polynomial_order, args.matfree)
    records = [
        TimeRecord(initial_run=index == 0, p=args.polynomial_order, matfree=args.matfree,
                   N=args.N, time=solver.solve_psi(True))
        for index in range(args.num_solves)
    ]

    if sweep.is_main_rank():
        sweep.save_records(args.out, records)

def eval_ns(Ns, p : int, matfree : bool, num_solves : int, max_ranks : int):
    """Time every resolution in Ns, one fresh process per point."""
    records = []
    for N in Ns:
        ranks = sweep.calc_ranks(p, N, max_ranks)
        point_args = (["-N", N, "-p", p, "-ns", num_solves]
                      + (["--matfree"] if matfree else []))
        records.extend(sweep.run_point(__file__, ranks, point_args, TimeRecord))
    return records

def _dofs_vs_time(records):
    """Average solve time per degrees-of-freedom value across a list of TimeRecords."""
    grouped = defaultdict(list)
    for r in records:
        grouped[r.dofs()].append(r.time)
    dofs = sorted(grouped)
    times = [np.mean(grouped[d]) for d in dofs]
    return np.array(dofs), np.array(times)

def plot_time_complexity(json_path, output_path="tex/time_complexity.pdf"):
    """
    Create a log-log plot of solve time vs degrees of freedom from a
    time_complexity_*.json results file.
    """
    import matplotlib.pyplot as plt

    import plot_utils
    plot_utils.apply_style()

    records = sweep.load_records(json_path, TimeRecord)

    series = [
        ("Assembled matrix, initial solve", False, True, '#004488', 'o', '-'),
        ("Assembled matrix, subsequent solves", False, False, '#004488', 's', '--'),
        ("Matrix free, initial solve", True, True, '#BB5566', 'o', '-'),
        ("Matrix free, subsequent solves", True, False, '#BB5566', 's', '--'),
    ]

    plt.figure(figsize=plot_utils.FIGURE_SIZE)

    for label, matfree, initial_run, color, marker, linestyle in series:
        subset = [r for r in records if r.matfree == matfree and r.initial_run == initial_run]
        if not subset:
            continue
        dofs, times = _dofs_vs_time(subset)
        plt.loglog(dofs, times, color=color, marker=marker, linestyle=linestyle,
                   linewidth=1.5, markersize=4, label=label)

        print(f"{label}: average log-log slope = {plot_utils.log_log_slope(dofs, times):.3f}")

    plt.xlabel(r'Degrees of freedom')
    plt.ylabel(r'Solve time [\unit{\second}]')
    plot_utils.finish_figure(output_path)

def main():
    parser = argparse.ArgumentParser(description='Get performance results for the psi solver')
    parser.add_argument('-p', '--polynomial_order', type=int, default=4)
    parser.add_argument('-ns', '--num_solves', type=int, default=2)
    parser.add_argument('-assd', '--max_dofs_assembled', type=int, default=3000000)
    parser.add_argument('-matfd', '--max_dofs_matfree', type=int, default=6000000)
    parser.add_argument('-n', '--num_resolutions', type=int, default=5)
    parser.add_argument('-ni', '--num_initial_solves', type=int, default=1)
    sweep.add_common_arguments(parser)
    sweep.add_point_arguments(parser)
    # Internal re-exec entry point - not for direct use.
    parser.add_argument('--matfree', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single_point:
        _run_point(args)
        return

    if args.plot:
        plot_time_complexity(args.plot)
        return

    min_dofs = args.ranks * 100000

    records = []
    for _ in range(args.num_initial_solves):

        for matfree, max_dofs in [(False, args.max_dofs_assembled), (True, args.max_dofs_matfree)]:

            Ns = sweep.resolutions_for_dofs(min_dofs, max_dofs, args.num_resolutions, args.polynomial_order)
            records.extend(eval_ns(Ns, args.polynomial_order, matfree, args.num_solves, args.ranks))

    sweep.save_records(f"time_complexity_{args.job_id}.json", records, indent=2)

if __name__ == '__main__':
    main()
