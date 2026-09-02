"""
Driver script to generate Module M14a figures (Paper Figs. 4-5 analogs).

Produces:
    - results/figures/fig4_trajectories_comparison.png (3D flight trajectories for all 5 methods)
    - results/figures/fig5_snapshots_tdpk.png
    - results/figures/fig5_snapshots_greedy.png
    - results/figures/fig5_snapshots_duelingdql.png
    - results/figures/fig5_snapshots_ppo.png
    - results/figures/fig5_snapshots_pktd3-td.png
"""

from uav_trajectory_rl.evaluation.figures_4_5 import (
    DEFAULT_METHODS,
    generate_fig4_trajectories,
    generate_fig5_snapshots,
)


def main():
    print("=" * 70)
    print("Generating Module M14a figures (Paper Figs. 4-5 analogs)...")
    print("=" * 70)

    # 1. Fig. 4 Trajectory comparison
    fig4_path = generate_fig4_trajectories()
    print(f"[OK] Fig. 4: {fig4_path} ({fig4_path.stat().st_size} bytes)")

    # 2. Fig. 5 Snapshots for each method
    for method_name in DEFAULT_METHODS:
        fig5_path = generate_fig5_snapshots(method_name)
        print(f"[OK] Fig. 5 ({method_name}): {fig5_path} ({fig5_path.stat().st_size} bytes)")

    print("=" * 70)
    print("All M14a figures successfully generated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
