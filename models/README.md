# Model registry

Repo này đã include các model nhỏ/gọn đủ để tái chạy những benchmark chính của nhánh motion/person-gate. Dataset, video demo, cache train và checkpoint trung gian vẫn để ngoài Git.

| File | Vai trò | Ghi chú |
| --- | --- | --- |
| `yolov5_lumi.pt` | YOLOv5 Lumi baseline/person detector | Dùng cho pipeline YOLO Lumi + motion và làm nguồn feature cho Silu grid. Không include ONNX/RKNN Lumi trong repo này. |
| `silu26_teacher_epoch008.pt` | Human grid teacher | Model Silu26 có kết quả tốt nhất trong nhóm heat-grid đã benchmark. |
| `student_mbv2_s32_epoch019_cached.pt` | Student distillation nhỏ | Model student epoch 19 đã benchmark; chưa đạt teacher nhưng là mốc thử nghiệm quan trọng. |
| `vww_domain_camera_best.pt` | Person crop classifier | Model VWW/PyTorch finetune trên crop camera-domain, dùng trong hướng motion proposal + classifier. |

Kiểm tra integrity:

```bash
cd motion_detection_handoff
sha256sum -c models/checksums.sha256
```

Các model không đưa vào repo:

- `*.rknn`: phần motion/person-gate trong repo này không chạy RKNN.
- `yolov5_lumi.onnx`: file lớn và không cần cho pipeline bàn giao hiện tại vì đã có `.pt`.
- YOLO26L/YOLOv8M probe checkpoint: chỉ dùng để khảo sát kiến trúc/cut-layer, chưa phải model chính trong benchmark cuối.
- Dataset, video demo, frame cache, teacher cache và checkpoint từng epoch.
