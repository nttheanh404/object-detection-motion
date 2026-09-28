import numpy as np

from object_motion.grid import connected_components, logits_to_probability, yolo_boxes_to_soft_grid


def test_soft_grid_and_components():
    target = yolo_boxes_to_soft_grid([(0.5, 0.5, 0.25, 0.5)], grid_w=8, grid_h=6)
    assert target.shape == (6, 8)
    assert 0.0 < float(target.max()) <= 1.0
    probability = logits_to_probability(np.zeros((6, 8), dtype=np.float32))
    assert np.allclose(probability, 0.5)
    components = connected_components(target, threshold=0.1, min_cells=1)
    assert components

