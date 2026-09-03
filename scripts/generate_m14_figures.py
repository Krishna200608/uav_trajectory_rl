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
    - M14c:
        - results/figures/fig7a_uav_position_density.png (UAV xy-position 2-D KDE across 10 flights)
        - results/figures/fig7b_altitude_and_user_density.png (UAV altitude 1-D KDE & user xy-position 2-D KDE)
    - M14d:
        - results/figures/fig8_user_sweep.png (Performance metrics vs. number of users k in [10..20])
    - M14e:
        - results/figures/fig9_speed_sweep.png (Performance metrics vs. user mobility speed in (2, 4, 6, 8, 10, 12) m/s)
"""

import time

from uav_trajectory_rl.evaluation.figures_4_5 import (
    DEFAULT_METHODS,
    generate_fig4_trajectories,
    generate_fig5_snapshots,
)
from uav_trajectory_rl.evaluation.figures_6 import (
    generate_fig6_realtime_curves,
)
from uav_trajectory_rl.evaluation.figures_7 import (
    generate_fig7a_uav_position_density,
    generate_fig7b_altitude_and_user_density,
)
from uav_trajectory_rl.evaluation.figures_8 import (
    generate_fig8_user_sweep,
)
from uav_trajectory_rl.evaluation.figures_9 import (
    generate_fig9_speed_sweep,
)


def main():
    print("=" * 70)
    print("Generating Module M14 evaluation figures...")
    print("=" * 70)

    # --- M14a: Fig. 4 Trajectory comparison ---
    print("\n[1/7] Generating Fig. 4: 3-D flight trajectory comparison...")
    fig4_path = generate_fig4_trajectories()
    print(f"[OK] Fig. 4: {fig4_path} ({fig4_path.stat().st_size} bytes)")

    # --- M14a: Fig. 5 Snapshots for each method ---
    print("\n[2/7] Generating Fig. 5: Time-slot snapshots for all 5 methods...")
    for method_name in DEFAULT_METHODS:
        fig5_path = generate_fig5_snapshots(method_name)
        print(f"[OK] Fig. 5 ({method_name}): {fig5_path} ({fig5_path.stat().st_size} bytes)")

    # --- M14b: Fig. 6 Real-time LoS and Rate curves ---
    print("\n[3/7] Generating Fig. 6: Real-time LoS probability & transmission rate...")
    fig6_path = generate_fig6_realtime_curves()
    print(f"[OK] Fig. 6: {fig6_path} ({fig6_path.stat().st_size} bytes)")

    # --- M14c: Fig. 7(a) UAV Position Density ---
    print("\n[4/7] Generating Fig. 7(a): UAV position kernel density (10 flights)...")
    fig7a_path = generate_fig7a_uav_position_density()
    print(f"[OK] Fig. 7(a): {fig7a_path} ({fig7a_path.stat().st_size} bytes)")

    # --- M14c: Fig. 7(b) Altitude & User Density ---
    print("\n[5/7] Generating Fig. 7(b): Altitude density & user position density (10 flights)...")
    fig7b_path = generate_fig7b_altitude_and_user_density()
    print(f"[OK] Fig. 7(b): {fig7b_path} ({fig7b_path.stat().st_size} bytes)")

    # --- M14d: Fig. 8 Sweep vs. Number of Users ---
    print("\n[6/7] Generating Fig. 8: Performance sweep vs. number of users k in [10..20]...")
    print("NOTE: Sweeping TDPK and Greedy over 11 k-values x 5 seeds; this may take several minutes.")
    t0 = time.time()
    fig8_path = generate_fig8_user_sweep(
        k_values=range(10, 21),
        sweep_seeds=range(5),
        reference_seeds=range(5),
    )
    duration = time.time() - t0
    print(f"[OK] Fig. 8: {fig8_path} ({fig8_path.stat().st_size} bytes, elapsed: {duration:.1f}s)")

    # --- M14e: Fig. 9 Sweep vs. User Mobility Speed ---
    print("\n[7/7] Generating Fig. 9: Performance sweep vs. user mobility speed v_mob in (2, 4, 6, 8, 10, 12) m/s...")
    print("NOTE: Sweeping all 5 methods over 6 speed points x 5 seeds; this may take several minutes.")
    t0 = time.time()
    fig9_path = generate_fig9_speed_sweep(
        speed_values=(2, 4, 6, 8, 10, 12),
        sweep_seeds=range(5),
        k=10,
    )
    duration = time.time() - t0
    print(f"[OK] Fig. 9: {fig9_path} ({fig9_path.stat().st_size} bytes, elapsed: {duration:.1f}s)")

    print("\n" + "=" * 70)
    print("All M14 figures successfully generated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
