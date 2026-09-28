# Ghi chú bàn giao

Repo này là bản rút gọn để người tiếp nhận có thể chạy lại và tiếp tục nghiên cứu mà
không phải đọc toàn bộ workspace robot_ws.

## Việc đã hoàn thành

- Có baseline MOG2 và detector robust trả soft motion grid.
- Đã thử human grid từ Lumi Silu10/Silu26, YOLO26 cut và distillation MobileNetV2.
- Đã thử hai hướng: human grid × motion và motion crop × person classifier.
- Có manifest domain-camera tách khỏi COCO, script đánh giá, video demo và summary JSON.
- Có baseline YOLO Lumi ONNX/RKNN cùng pipeline decode/preprocess để đối chiếu phần cứng.

## Việc cần làm tiếp

- Chọn operating point theo recall người và chi phí false alarm thực tế.
- Bổ sung ground truth motion độc lập; hiện nhiều bảng dùng YOLO pseudo-label.
- Đánh giá lại distillation sau khi sửa target/loss hoặc chọn student lớn hơn.
- Tách preprocessing, inference và postprocess trong từng benchmark.
- Đóng gói model qua artifact store/Git LFS thay vì đưa checkpoint lớn vào Git thường.

