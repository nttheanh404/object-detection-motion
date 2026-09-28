# Inventory nguồn trong repo bàn giao

## API ổn định

- `src/object_motion/motion/robust_detector.py`
- `src/object_motion/motion/mog2.py`
- `src/object_motion/grid/targets.py`
- `src/object_motion/grid/postprocess.py`
- `tools/benchmark_motion.py`
- `tools/check_environment.py`

## Script thí nghiệm giữ nguyên

Xem `legacy/README.md`. Các script này được giữ để người nhận đối chiếu với những run
cũ, không coi là API ổn định.

## Artifact ngoài Git

Dataset, video, checkpoint, ONNX/RKNN và các output lớn không nằm trong repo. Bản đồ
đường dẫn và tên file cũ nằm ở:

`/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_file_inventory_2026-09-28.md`

Repo này chỉ giữ README/manifest reference để có thể push GitHub mà không đưa khoảng
5 GB artifact vào lịch sử Git.

