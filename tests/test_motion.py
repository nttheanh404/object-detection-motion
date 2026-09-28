import numpy as np

from object_motion import MOG2MotionDetector, RobustMotionDetector


def test_motion_detectors_return_finite_scores():
    first = np.zeros((96, 128, 3), dtype=np.uint8)
    second = first.copy()
    second[32:64, 40:72] = 255
    robust = RobustMotionDetector(grid_width=16, grid_height=12)
    assert robust.process(first).shape == (12, 16)
    score = robust.process(second)
    assert score.shape == (12, 16)
    assert np.isfinite(score).all()
    assert float(score.max()) >= 0.0

    mog = MOG2MotionDetector()
    assert mog.process_grid(first, 16, 12).shape == (12, 16)
    assert np.isfinite(mog.process_grid(second, 16, 12)).all()

