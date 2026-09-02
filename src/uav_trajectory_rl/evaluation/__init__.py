"""
Evaluation and plotting suite subpackage (Module M14).
"""

from uav_trajectory_rl.evaluation.figures_4_5 import (
    generate_fig4_trajectories,
    generate_fig5_snapshots,
)
from uav_trajectory_rl.evaluation.harness import (
    EpisodeLog,
    MethodSpec,
    get_method_specs,
    run_batch,
    run_episode,
)

__all__ = [
    "MethodSpec",
    "EpisodeLog",
    "get_method_specs",
    "run_episode",
    "run_batch",
    "generate_fig4_trajectories",
    "generate_fig5_snapshots",
]
