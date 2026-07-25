"""kinescore -- physics-plausibility benchmark for AI-generated robot video.

Reads a robot's joint configuration from pixels with a frozen vision backbone,
projects it onto an exact forward-kinematics model, and measures analytic
physics residuals. Robot-agnostic core; Franka and GR-1 plugins ship.

The top level stays import-light: pull contracts from :mod:`kinescore.core`,
robots from :mod:`kinescore.robots`, metrics from :mod:`kinescore.metrics`.
"""
__version__ = "0.1.0"
