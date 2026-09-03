"""
Driver script to generate Module M14 evaluation figures.

Covers:
    - M14a:
        - results/figures/fig4_trajectories_comparison.png (3D flight trajectories for all 5 methods)
        - results/figures/fig5_snapshots_tdpk.png
        - results/figures/fig5_snapshots_greedy.png
        - results/figures/fig5_snapshots_duelingdql.png
        - results/figures/fig5_snapshots_ppo.png
        - results/figures/fig5_snapshots_pktd3-td.png
    - M14b:
        - results/figures/fig6_realtime_curves.png (LoS probability & transmission rate vs. time slot)
"""

from uav_trajectory_rl.evaluation.figures_4_5 import (
    DEFAULT_METHODS,
    generate_fig4_trajectories,
    generate_fig5_snapshots,
)
from uav_trajectory_rl.evaluation.figures_6 import (
    generate_fig6_realtime_curves,
)


def main():
    print("=" * 70)
    print("Generating Module M14 evaluation figures...")
    print("=" * 70)

    # --- M14a: Fig. 4 Trajectory comparison ---
    print("\n[1/3] Generating Fig. 4: 3-D flight trajectory comparison...")
    fig4_path = generate_fig4_trajectories()
    print(f"[OK] Fig. 4: {fig4_path} ({fig4_path.stat().st_size} bytes)")

    # --- M14a: Fig. 5 Snapshots for each method ---
    print("\n[2/3] Generating Fig. 5: Time-slot snapshots for all 5 methods...")
    for method_name in DEFAULT_METHODS:
        fig5_path = generate_fig5_snapshots(method_name)
        print(f"[OK] Fig. 5 ({method_name}): {fig5_path} ({fig5_path.stat().st_size} bytes)")

    # --- M14b: Fig. 6 Real-time LoS and Rate curves ---
    print("\n[3/3] Generating Fig. 6: Real-time LoS probability & transmission rate...")
    fig6_path = generate_fig6_realtime_curves()
    print(f"[OK] Fig. 6: {fig6_path} ({fig6_path.stat().st_size} bytes)")

    print("\n" + "=" * 70)
    print("All M14 figures successfully generated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
