# Checklist tái lập

1. Mount HDD16T và kiểm tra các thư mục trong `data/README.md`.
2. Cài package editable và chạy `pytest -q`.
3. Chọn manifest cố định; không trộn COCO với domain camera nếu đang đánh giá domain camera.
4. Ghi model path, input size, grid size, threshold, backend CPU/GPU và commit hash.
5. Chạy motion-only trước để đo decode/preprocess/motion/postprocess.
6. Chạy human-grid hoặc YOLO gate trên cùng frame list.
7. Nếu dùng crop classifier, lưu số candidate/frame, số candidate có bbox người,
   false alarm, miss và thời gian từng stage.
8. Lưu JSON raw cùng Markdown summary; video chỉ là artifact kiểm tra trực quan.

Không coi video preview là ground truth. Các kết quả YOLO-grid trong báo cáo cũ là
pseudo-label comparison và cần được kiểm tra lại bằng label domain camera khi có bộ
đánh giá độc lập.

