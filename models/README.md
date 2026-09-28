# Model registry

Model lớn không nằm trong Git repository. Bản đã dùng trong thí nghiệm nằm ở:

- YOLOv5 Lumi: `/home/ai5070/babyalpha/theanh/heatmap/yolov5_lumi.pt` và `.onnx`
- YOLOv8m Lumi: `benchmark_results/motion_detection_datasets/yolov8m_lumi.pt`
- YOLO26l Lumi: `benchmark_results/motion_detection_datasets/yolo26l_lumi.pt`
- Grid distillation epoch 19: `.../run_distill_mbv2_s32_silu26_v1/epoch_019_cached.pt`
- Silu26 teacher epoch 8: `.../run_silu26_finetune_from_silu10_v1/epoch_008.pt`
- VWW domain-camera: `.../person_crop_classifier_runs/vww_pytorch_domain_camera_v1/best.pt`

Khi bàn giao sang máy khác, copy model vào thư mục ngoài Git rồi truyền đường dẫn qua
CLI/config. Không commit checkpoint hoặc video vào repo chính; dùng Git LFS/artifact store
nếu cần chia sẻ chúng.

