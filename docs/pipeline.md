# Kiến trúc pipeline

```text
frame/video
   │
   ├── resize/luma/background model
   │       └── soft motion grid [H×W]
   │
   ├── YOLO Lumi hoặc human-grid model
   │       └── human probability grid / person boxes
   │
   ├── gate = motion × human probability
   │       └── connected components / crop proposals
   │
   └── classifier hoặc cloud verification
```

## Motion proposal

`MOG2MotionDetector` là baseline đơn giản, dùng background subtraction, morphology,
persistence và dynamic-noise suppression. `RobustMotionDetector` là bản nhiều tầng:

- chuẩn hóa luma và texture;
- bù thay đổi sáng/tối toàn ảnh;
- phase correlation cho camera shift nhỏ;
- kết hợp photo/texture/temporal evidence;
- loại component nhỏ;
- smoothing nhanh khi motion xuất hiện, chậm khi motion biến mất;
- giảm ảnh hưởng của pattern lặp và global event.

Đầu ra là score float `[0, 1]`, không phải mask nhị phân. Ngưỡng chỉ áp dụng ở bước
post-process để dễ sweep.

## Human grid

Target từ label YOLO là tỷ lệ diện tích bbox phủ lên từng cell. `gamma < 1` làm mềm
border cell; đây là target mềm, không phải IoU của cả bbox. Grid stride 8 (`72×44` với
input `576×352`) là điểm cân bằng đã được thử; stride 4 chi tiết hơn nhưng nhiễu hơn,
stride 16 rẻ hơn nhưng dễ bỏ sót người nhỏ.

## Đánh giá

- Motion-only cần label motion theo frame; nếu không chỉ báo cáo hành vi và latency.
- Human-grid so với pseudo-label YOLO cần báo cáo person hit, image recall, cell
  precision, FP cells/frame và negative false trigger.
- Crop/classifier cần đánh giá trên cùng candidate motion regions, không trộn với full
  frame metric.
- Không dùng `pred_pos_frac` hoặc số active cell đơn độc làm accuracy.

