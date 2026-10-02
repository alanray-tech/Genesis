"""Teleoperate Franka cloth contact with the CGQ MinCoo rigid backend.

This entry point provides the same scene, GUI controls, headless checks, and
performance overlay as ``franka_cloth_cube_teleop.py`` while selecting
``rigid/dynamics_backend = "cgq_mincoo"``.
"""

from franka_cloth_cube_teleop import main

if __name__ == "__main__":
    main(rigid_backend="cgq_mincoo")
