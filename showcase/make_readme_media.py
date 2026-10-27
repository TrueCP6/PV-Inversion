"""README media: GIFs cut from the two videos, a hero render, and figures from the thesis PDF.

    python showcase/make_readme_media.py DATA_DIR MEDIA_DIR [diagnostic] [prognostic] [figures]

With no parts named, all three are made. MEDIA_DIR must already hold the videos the parts
cut from (diagnostic_solver.mp4, prognostic_solver.mp4).
"""
import argparse
import os
import subprocess
import matplotlib.pyplot as plt
import render_diagnostic
import render_prognostic
import thesis_figures
import videokit as vk

PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "thesis.pdf")


def gif(video, out, start, duration, width=960, fps=10, speed=1.0):
    """A palette-optimised GIF of [start, start + duration) of `video`."""
    filters = (f"setpts={1 / speed}*PTS,fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
               "[a]palettegen=max_colors=160:stats_mode=diff[p];"
               "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", str(duration),
                    "-i", video, "-vf", filters, out], check=True)
    print(f"{out}: {os.path.getsize(out) / 1e6:.1f} MB")


def scene_start(scenes, index):
    return sum(s.duration for s in scenes[:index])


def diagnostic(data_dir, media_dir):
    # Scene timings come from the renderer itself, so the cut follows any re-timing
    d = render_diagnostic.load(data_dir)
    figs = thesis_figures.load(PDF, ["scaling", "convergence"])
    scenes = render_diagnostic.scenes(d, figs)
    gif(os.path.join(media_dir, "diagnostic_solver.mp4"), os.path.join(media_dir, "diagnostic_race.gif"),
        scene_start(scenes, 2) + 0.4, scenes[2].duration - 0.7)

    # Hero: the intrusion's low and the surface winds it drives, as a wide banner
    m = render_diagnostic.meshes(d)
    stage = render_diagnostic.output_stage(d, m, (2400, 1000), tropopause=True)
    plt.imsave(os.path.join(media_dir, "hero.png"), stage.shot(-62, 14, 3900, 5.0))


def prognostic(data_dir, media_dir):
    run = render_prognostic.Run(data_dir)
    figs = thesis_figures.load(PDF, ["evolution"])
    figs["evolution_last"] = render_prognostic.last_panel(figs["evolution"])
    scenes = render_prognostic.scenes(run, figs)
    gif(os.path.join(media_dir, "prognostic_solver.mp4"),
        os.path.join(media_dir, "prognostic_evolution.gif"),
        scene_start(scenes, 2) + 0.4, scenes[2].duration - 0.7, speed=1.5)


def figures(media_dir):
    for name, img in thesis_figures.load(PDF, ["scaling", "convergence"]).items():
        plt.imsave(os.path.join(media_dir, f"{name}.png"), img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("media_dir")
    parser.add_argument("parts", nargs="*", default=["diagnostic", "prognostic", "figures"])
    args = parser.parse_args()
    vk.setup_style()
    if "diagnostic" in args.parts:
        diagnostic(args.data_dir, args.media_dir)
    if "prognostic" in args.parts:
        prognostic(args.data_dir, args.media_dir)
    if "figures" in args.parts:
        figures(args.media_dir)


if __name__ == "__main__":
    main()
