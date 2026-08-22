"""kinescore -- physics-plausibility judging for generated robot video.

A frozen vision backbone and a trained head read 3-D keypoints from frames;
five analytic detectors score physics violations on those keypoints.

The top level stays import-light: pull contracts from :mod:`kinescore.core`,
robots from :mod:`kinescore.robots`, detectors from :mod:`kinescore.violations`.
"""
__version__ = "0.2.0"
