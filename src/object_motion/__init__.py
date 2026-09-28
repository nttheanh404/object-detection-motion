"""Fixed-camera motion and human-focused grid detection."""

from .motion.robust_detector import MotionConfig, RobustMotionDetector
from .motion.mog2 import MOG2MotionDetector

__all__ = ["MotionConfig", "RobustMotionDetector", "MOG2MotionDetector"]

