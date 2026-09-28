# Human Object Motion Detection

Repo bàn giao cho pipeline phát hiện chuyển động ở camera cố định và lọc vùng có
người. Repo chứa code, cấu hình, test và báo cáo; ảnh/video/label/checkpoint lớn được
giữ ngoài Git.

Pipeline có hai nhánh độc lập:

1. **Motion proposal**: MOG2 baseline hoặc RobustMotionDetector tạo soft motion grid.
2. **Human gate**: YOLO Lumi hoặc model grid human-focus xác định vùng có người; có thể
   kết hợp với motion để tạo candidate crop trước khi gửi lên classifier/cloud.

Không gọi motion score là person detection. Motion chỉ trả lời vùng nào đang thay đổi;
human grid/classifier mới quyết định vùng đó có khả năng chứa người.

## Cài đặt

```bash
cd motion_detection_handoff
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Nếu chạy grid/classifier bằng PyTorch/ONNX Runtime:

```bash
python -m pip install -e ".[grid]"
```

Kiểm tra môi trường và đường dẫn dữ liệu:

```bash
python tools/check_environment.py
pytest -q
```

## Chạy motion baseline

```bash
python tools/benchmark_motion.py \
  --input /path/to/video.mp4 \
  --outdir outputs/mot17_demo \
  --grid-w 32 --grid-h 24 --threshold 0.30
```

Lệnh tạo `summary.json` và video side-by-side Robust/MOG2. Dùng `--no-video` nếu chỉ
muốn đo tốc độ và giảm I/O.

## Dữ liệu và model

Xem [data/README.md](data/README.md) và [models/README.md](models/README.md). Bản đầy đủ
của inventory cũ nằm ở [../benchmark_results/motion_detection_file_inventory_2026-09-28.md](../benchmark_results/motion_detection_file_inventory_2026-09-28.md).

## Kết quả đã biết

Trên validation 3.000 ảnh, model Silu26 teacher epoch 8 ở threshold 0.60 đạt person
hit@1 full 81,0%, image recall 91,5%, cell precision 81,0%; threshold 0.70 tăng cell
precision lên 91,2% nhưng image recall giảm còn 86,1%. Student distill MBV2 S32 epoch
19 chưa đạt mức tương đương teacher và cần xem lại trước khi dùng làm model chính.

Các số liệu chi tiết và giới hạn đánh giá nằm trong [docs/benchmark_results.md](docs/benchmark_results.md).

## Nguyên tắc bàn giao

- Không commit dữ liệu camera, video, checkpoint hoặc file `.onnx.data` vào Git.
- Mỗi benchmark phải ghi model, manifest, threshold, input size, backend và commit hash.
- Giữ riêng motion proposal và human gate để có thể thay thế từng module.
- Khi so sánh chất lượng, dùng cùng manifest và cùng định nghĩa positive/negative.

