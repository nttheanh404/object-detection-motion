# Kết quả nền đã bàn giao

Nguồn số liệu: `benchmark_results/motion_detection_datasets/silu10_grid_train/grid_detection_metrics_recent/metrics.md`.

Validation gồm 3.000 ảnh. GT cell là cell có tâm nằm trong bbox người; `core` là bbox
co lại 50%. Các số dưới đây là đối chiếu với pseudo-label YOLO, chưa phải ground truth
độc lập.

| Model | Threshold | Person hit@1 full | Image recall | Cell precision | FP cells/frame | Negative false trigger |
|---|---:|---:|---:|---:|---:|---:|
| Silu26 teacher epoch 8 | 0.60 | 81,0% | 91,5% | 81,0% | 36,1 | 22,6% |
| Silu26 teacher epoch 8 | 0.70 | 73,2% | 86,1% | 91,2% | 11,9 | 10,7% |
| Student MBV2 S32 epoch 19 | 0.60 | 43,9% | 47,3% | 76,5% | 40,5 | 4,4% |
| Student MBV2 S32 epoch 19 | 0.70 | 33,0% | 37,5% | 87,7% | 13,3 | 1,4% |

Diễn giải: tăng threshold giúp precision và giảm false trigger nhưng làm recall giảm.
Mục tiêu lọc người cần chọn điểm operating theo chi phí bỏ sót, không chọn theo cell
precision riêng lẻ. Distillation MBV2 S32 hiện chưa giữ được recall của teacher.

Các benchmark video, classifier và YOLO/Lumi nằm trong các thư mục báo cáo cũ được liệt
kê trong `benchmark_results/motion_detection_file_inventory_2026-09-28.md`.

