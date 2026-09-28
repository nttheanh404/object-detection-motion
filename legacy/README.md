# Historical experiment scripts

Các script trong thư mục này được chép từ workspace thí nghiệm để truy nguyên kết quả
đã công bố. Chúng giữ nguyên CLI và một số đường dẫn tuyệt đối của máy cũ, nên không
phải API ổn định của package mới.

- `motion/`: detector cổ điển, RobustMotionDetector và YOLO Lumi + motion.
- `grid/`: train/benchmark Silu10, Silu26, YOLO cut và MobileNetV2 distillation.
- `classifier/`: tạo crop dataset, chuyển VWW và train person classifier.
- `evaluation/`: so sánh candidate windows/classifier với pseudo-label YOLO.

Khi tiếp tục phát triển, nên sửa script thành module trong `src/` và nhận đường dẫn qua
CLI/config thay vì dùng đường dẫn tuyệt đối.

