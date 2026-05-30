# Teacher-Student NST A/B/C Segmentation Pipeline

## Ý tưởng chỉnh

- `6v1_unet_model.py` đã đổi sang pipeline **Teacher-Student**.
- Teacher mạnh hơn Student bằng `teacher_base_filters=48`, Student nhẹ hơn bằng `student_base_filters=24`.
- Teacher học hard label `A/B/C/Edge` trước.
- Student học hard label và soft output của Teacher để giữ hình thái A/B tốt hơn, đồng thời channel C vẫn được weight cao để giữ mạnh vùng overlap.
- Edge auxiliary channel chỉ dùng để học biên, output apply chính vẫn là A, B, C.
- Code tự resume từ:
  - `results/checkpoints_teacher/epoch_*.keras`
  - `results/checkpoints_student/epoch_*.keras`
- Sau train, code test `train`, `val`, `test`, xuất JSON/CSV trong `results/metric_reports`.
- Model tốt nhất để apply được copy thành `results/best_for_apply.keras`.

## Thứ tự chạy

```bash
python 1v1_create_folder.py
python 2v1_remove_subfolder.py          # nếu single_chromosomes_raw đang có nhiều folder con
python 2v1_prepare_single_chromosomes.py
python 3v1_generate_synthetic_masks.py
python 4v1_preprocess_to_256.py
python 5v1_split_data.py
python 6v1_unet_model.py --epochs 200 --batch-size 64 --min-acc 0.85 --auto-retrain-rounds 1
python 7v1_predict_real_overlap.py
```

## Output sau train

```text
results/best_teacher.keras
results/best_student.keras
results/best_for_apply.keras
results/best_model_summary.json
results/metric_reports/teacher_train_metrics.json
results/metric_reports/teacher_val_metrics.json
results/metric_reports/teacher_test_metrics.json
results/metric_reports/student_train_metrics.json
results/metric_reports/student_val_metrics.json
results/metric_reports/student_test_metrics.json
```

## Output sau apply vào overlap_raw

```text
results/real_predictions/masks_A_original_size/
results/real_predictions/masks_B_original_size/
results/real_predictions/masks_C_original_size/
results/real_predictions/overlays_original_size/
results/real_predictions/contours_original_size/
results/real_predictions/visualizations_256/
results/real_predictions/separated_chromosomes/
```

Trong `separated_chromosomes` sẽ có:

```text
<ten_anh_goc>_A.png
<ten_anh_goc>_B.png
<ten_anh_goc>_A_mask.png
<ten_anh_goc>_B_mask.png
```

Vùng C được đưa vào cả A và B để giữ full hình thái. Phần overlap/bị đè được fill bằng inpainting từ vùng visible của chính NST đó.

## Nếu Colab T4 bị OOM

Default giữ đúng yêu cầu `batch_size=64`. Nếu OOM, chạy lại:

```bash
python 6v1_unet_model.py --epochs 200 --batch-size 32 --min-acc 0.85 --auto-retrain-rounds 1
```

hoặc:

```bash
python 6v1_unet_model.py --epochs 200 --batch-size 16 --min-acc 0.85 --auto-retrain-rounds 1
```

## Lưu ý checkpoint

Không cần train lại từ đầu nếu đã có checkpoint. Script tự ưu tiên checkpoint epoch mới nhất rồi mới đến best model.

Muốn train lại từ đầu thì thêm:

```bash
python 6v1_unet_model.py --force-restart
```
