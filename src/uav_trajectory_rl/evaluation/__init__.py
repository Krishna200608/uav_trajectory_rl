"""
Evaluation and plotting suite subpackage (Module M14).
"""

from uav_trajectory_rl.evaluation.figures_4_5 import (
    generate_fig4_trajectories,
    generate_fig5_snapshots,
)
from uav_trajectory_rl.evaluation.figures_6 import (
    METHOD_COLORS,
    generate_fig6_realtime_curves,
)
from uav_trajectory_rl.evaluation.figures_7 import (
    generate_fig7a_uav_position_density,
    generate_fig7b_altitude_and_user_density,
)
from uav_trajectory_rl.evaluation.figures_8 import (
    _compute_dte,
    generate_fig8_user_sweep,
)
from uav_trajectory_rl.evaluation.figures_9 import (
    generate_fig9_speed_sweep,
    speed_to_range,
)
from uav_trajectory_rl.evaluation.summary_table import (
    METHOD_BEHAVIOR,
    generate_summary_table,
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
    "generate_fig6_realtime_curves",
    "generate_fig7a_uav_position_density",
    "generate_fig7b_altitude_and_user_density",
    "generate_fig8_user_sweep",
    "generate_fig9_speed_sweep",
    "generate_summary_table",
    "speed_to_range",
    "_compute_dte",
    "METHOD_COLORS",
    "METHOD_BEHAVIOR",
]
