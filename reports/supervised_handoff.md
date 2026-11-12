# Bàn giao Gradient Boosting

Phần supervised được triển khai trên `feature/gradient-boosting`, từ nền tích hợp anomaly `53ec200` và EDA train `9c0821c` (merge nền `c9c58cd`). Dùng bốn `FEATURE_COLUMNS`, human=0/bot=1 và split chung 60/20/20, seed 42. Không dùng `user_id` hoặc `label` làm đầu vào dự đoán.

## Tái tạo

Tại thư mục gốc, dùng môi trường cài từ `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe -m src.supervised
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Bash sau khi kích hoạt môi trường: `python -m src.supervised`. Hoặc Run All `notebooks/03_supervised_baseline.ipynb` để tạo cả biểu đồ. Giữ nguyên `dataset/processed/split.csv`; hàm chia tập sẽ báo lỗi nếu ID không khớp. Trên dữ liệu hiện có: train 22.462, validation 7.488 dòng. Split SHA-256: `d162338f765ea03ce4081fc8b3fc4fa65048fa9c544bb6f4e12f10371f6ffe19`.

Mỗi lần chạy ghi đè các model/báo cáo với `run_id` mới. Muốn giữ một thí nghiệm, sao lưu cùng model, CSV và manifest trước khi chạy tiếp. Thời gian chạy và `run_id` thay đổi; cấu hình, ngưỡng và dự đoán được kiểm tra tái lập theo seed trong môi trường hiện có. Metadata ghi phiên bản thư viện thực tế; dùng cùng phiên bản khi tái tạo hoặc tải joblib.

## Kết quả validation

| Model / ngưỡng | Precision bot | Recall bot | F1 bot | Macro-F1 | AP | ROC-AUC | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gradient Boosting / 0,5 | 0,7766 | 0,6294 | 0,6953 | 0,7822 | 0,7876 | 0,8615 | 0,8169 |
| Gradient Boosting / F1 validation | 0,6851 | 0,7336 | 0,7085 | 0,7780 | 0,7876 | 0,8615 | 0,7997 |

Chọn cấu hình theo AP validation từ lưới **100/200 cây × độ sâu 2/3/4**, learning rate=0,1, min_samples_leaf=5, subsample=1, seed=42. AP bằng nhau giữ cấu hình đầu tiên trong lưới. Cấu hình chọn: **200 cây, max_depth=4**. Chọn ngưỡng F1 bot trên validation, giữ ngưỡng thấp nhất khi bằng nhau: **0,36915560905621475**. Dùng `score >= threshold`.

Median imputer chỉ fit trên train; cây không cần scaler/log1p. Cột hoàn toàn thiếu vẫn được giữ, điền 0. Không refit trên train+validation sau chọn ngưỡng. Confusion matrix theo `[[TN, FP], [FN, TP]]`: Gradient Boosting ngưỡng F1 `[[4165, 838], [662, 1823]]`.

Các số validation sau tìm kiếm có thể lạc quan. Notebook 04 hiện giữ lịch sử test đã được quan sát theo nhánh importance-analysis (commit `951a657`) và ghi lần đánh giá mới ngày 10/11/2026 với model chính thức. Kết quả tạm AUC 0,863 thuộc cấu hình 300 cây/max_depth=5, được giữ trong `reports/history/2026-11-09_interim_metrics.csv`. Lần chạy supervised này chỉ chọn cấu hình/ngưỡng trên validation; không dùng test để điều chỉnh tiếp.

## Tệp và giao diện model

- `models/gradient_boosting.joblib`: dictionary chứa pipeline tại khóa `model`; nằm ngoài Git theo quy tắc hiện có.
- `reports/supervised_validation_metrics.csv`: cả ngưỡng mặc định và ngưỡng F1 validation, chỉ số, confusion matrix, thời gian fit/score, `run_id`, split hash.
- `reports/supervised_validation_search.csv`: toàn bộ sáu cấu hình và cấu hình được chọn.
- `reports/supervised_validation_predictions.csv`: ID, nhãn thật, score, dự đoán mặc định/đã chọn, ngưỡng và `run_id`; chỉ validation.
- `reports/supervised_run.json`: cấu hình đầy đủ, quy tắc chọn, ngưỡng, đặc trưng, seed, phiên bản Python/NumPy/pandas/scikit-learn/joblib, thời gian, hash nguồn/split/mã nguồn. Dùng tệp này để lấy số liệu chính xác của lần chạy mới nhất.
- `reports/figures/supervised_validation_diagnostics.png`: Precision–Recall và confusion matrix từ notebook 03.

Ví dụ tích hợp vào notebook 04 sau khi có tập đánh giá được phép sử dụng:

```python
import joblib
import numpy as np
from src.config import FEATURE_COLUMNS, MODELS_DIR
from src.evaluate import evaluate_predictions
from src.train import bot_probabilities

bundle = joblib.load(MODELS_DIR / "gradient_boosting.joblib")
assert bundle["feature_columns"] == list(FEATURE_COLUMNS)
X = evaluation_rows[list(FEATURE_COLUMNS)]
scores = bot_probabilities(bundle["model"], X)
predictions = (scores >= bundle["threshold"]).astype(np.int64)
metrics = evaluate_predictions(evaluation_rows["label"], predictions, scores)
```

Đối chiếu hash nguồn/split với `bundle["provenance"]`, `run_id` giữa model và báo cáo trước tích hợp. Score luôn là xác suất của lớp `classes_ == 1`, kể cả trường hợp thứ tự cột thay đổi.

## Kiểm tra tích hợp

Phạm vi hiện tại gồm Isolation Forest, One-Class SVM và Gradient Boosting. Notebook 04 tải lại model để đánh giá mà không fit lại. Dữ liệu nguồn và split dùng chung được giữ nguyên. `evaluate_predictions()` trả AP/ROC-AUC là `None` khi truth chỉ có một lớp; Macro-F1 luôn tính cả hai nhãn. Hàm trả cả `tn/fp/fn/tp`, `confusion_matrix` và `flagged_share` cho các notebook dùng chung.

`src/comparison.py` so sánh ba mô hình, dùng `bot_probabilities()` cho Gradient Boosting và giữ `model.predict()` ở ngưỡng mặc định. Trước khi đánh giá test, nó kiểm tra schema, phiên bản scikit-learn, hash nguồn/split, run_id và đối chiếu AP validation với bundle. `reports/evaluation_run.json` ghi SHA-256 từng model, cấu hình/ngưỡng và hash bảng metrics. Các ca hồi quy bao gồm model sai nguồn/split, thiếu file, sai schema/phiên bản và khác AP validation.

Tái tạo đầy đủ: Run All notebook 01 → 02 → 03 → 04 sau khi tạo môi trường theo README. Mỗi lần chạy ghi đè bộ model/báo cáo hiện tại với run_id mới; giữ cả model và báo cáo nếu cần lưu một thí nghiệm. Model sinh ra vẫn được Git bỏ qua. Các mốc lịch sử 29/10–09/11 được giữ theo ghi chép cũ, không coi là holdout chưa quan sát; notebook 04 ghi lại lần đánh giá mới với model chính thức.

Nền tích hợp đã được merge vào `main`. Phạm vi hiện tại giữ ba mô hình theo đề bài; kết quả kiểm thử tự động không thay thế review chéo của thành viên.
