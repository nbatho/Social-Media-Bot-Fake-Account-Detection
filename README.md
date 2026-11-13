# Social Media Bot & Fake Account Detection

Phân loại tài khoản Twitter thành **human** và **bot** dựa trên bốn đặc trưng hồ sơ. Dự án dùng **Isolation Forest** và **One-Class SVM** để phát hiện bất thường, đối chiếu với **Gradient Boosting Classifier**, sau đó phân tích mức độ quan trọng của từng đặc trưng.

## 1. Nội dung bốn notebook

Chạy theo thứ tự **01 → 02 → 03 → 04**, sử dụng cùng môi trường Python.

| Notebook | Nội dung |
| --- | --- |
| [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb) | Kiểm tra dữ liệu, giá trị thiếu và ID trùng; tạo bốn đặc trưng; chia tập; khảo sát phân phối và tương quan Spearman trên train. |
| [02_anomaly_detection.ipynb](notebooks/02_anomaly_detection.ipynb) | Huấn luyện Isolation Forest và One-Class SVM; chọn cấu hình SVM theo AP validation; so sánh ngưỡng mặc định và ngưỡng F1 validation. |
| [03_supervised_baseline.ipynb](notebooks/03_supervised_baseline.ipynb) | Huấn luyện Gradient Boosting; chọn cấu hình và ngưỡng trên validation; lưu mô hình, chỉ số và biểu đồ. |
| [04_evaluation.ipynb](notebooks/04_evaluation.ipynb) | Tải ba mô hình đã lưu; so sánh validation/test; vẽ Precision–Recall, ma trận nhầm lẫn và phân tích feature importance. |

Các hàm dùng chung nằm trong `src/`. Notebook 04 chỉ tải và đánh giá mô hình, không huấn luyện lại.

## 2. Dữ liệu và đặc trưng

Nguồn: [Kaggle — Twitter Bots Accounts](https://www.kaggle.com/datasets/davidmartngutirrez/twitter-bots-accounts), tác giả David Martín Gutiérrez.

File đầu vào: `dataset/twitter_human_bots_dataset.csv`. Theo output hiện lưu trong notebook 01:

| Thuộc tính | Giá trị |
| --- | --- |
| Số tài khoản | 37.438 |
| Số cột nguồn | 20, gồm một cột index không tên |
| Human | 25.013 (66,81%) |
| Bot | 12.425 (33,19%) |
| ID trùng hoặc có nhãn mâu thuẫn | 0 |
| Bio rỗng | 7.256 |
| Bio chứa đúng chuỗi `None` | 1 |

Quy ước nhãn: **`human = 0`, `bot = 1`**. Cột `friends_count` biểu thị số tài khoản đang theo dõi.

| Đặc trưng | Cách tính |
| --- | --- |
| `follower_following_ratio` | `followers_count / max(friends_count, 1)` |
| `posting_frequency` | `statuses_count / max(account_age_days, 1)` |
| `account_age_days` | Tuổi tài khoản theo dữ liệu nguồn, tính bằng ngày |
| `bio_length` | Độ dài chuỗi `description`, gồm cả khoảng trắng và xuống dòng |

Pipeline dừng nếu thiếu cột bắt buộc, ID rỗng/trùng hoặc nhãn không hợp lệ. Giá trị số âm hoặc không đọc được được chuyển thành thiếu. Mẫu số bằng 0 được thay bằng 1; mẫu số thiếu vẫn cho kết quả thiếu. Bio rỗng có độ dài 0; chuỗi `None` và `NA` được giữ nguyên khi đọc bằng `load_profiles()`.

Notebook 01 đọc sơ bộ bằng pandas mặc định để đối chiếu cách nhận diện bio thiếu. Bảng đặc trưng chính thức được tạo bằng `run_pipeline()`, sử dụng quy tắc giữ nguyên văn bản nêu trên. Kết quả hiện lưu có **37.438 dòng × 6 cột**, không có đặc trưng thiếu.

Chỉ bốn đặc trưng trên được đưa vào mô hình. `user_id` dùng để định danh; `label` dùng để huấn luyện có giám sát và đánh giá.

## 3. Chia tập và huấn luyện

Dữ liệu được chia theo tài khoản, phân tầng theo nhãn với **seed 42**:

| Tập | Tỷ lệ | Số tài khoản |
| --- | --- | --- |
| Train | 60% | 22.462 |
| Validation | 20% | 7.488 |
| Test | 20% | 7.488 |

Lần đầu chạy, `split_features()` lưu phân hoạch vào `dataset/processed/split.csv`. Những lần sau dùng lại file này; nếu tập ID không khớp dữ liệu hiện tại, hàm báo lỗi. Giữ `split.csv` để các lần chạy sử dụng cùng phân hoạch.

| Mô hình | Tiền xử lý | Cấu hình |
| --- | --- | --- |
| Isolation Forest | Điền thiếu bằng trung vị | 300 cây, `contamination="auto"`, seed 42 |
| One-Class SVM | Điền thiếu bằng trung vị → `log1p` → StandardScaler | Kernel RBF; thử `nu ∈ {0.05, 0.1, 0.2}` và `gamma ∈ {"scale", 0.05, 1.0}` |
| Gradient Boosting | Điền thiếu bằng trung vị | Thử 100/200 cây × độ sâu 2/3/4; learning rate 0,1; tối thiểu 5 mẫu/lá; seed 42 |

Tiền xử lý chỉ được fit trên train. Hai detector được huấn luyện trên toàn bộ tập train, không sử dụng nhãn train. Gradient Boosting sử dụng nhãn train để học phân loại.

Cấu hình SVM và Gradient Boosting được chọn theo **Average Precision (AP) validation cao nhất**. Sau đó, mỗi mô hình được chọn ngưỡng tối đa **F1 lớp bot trên validation**. Các bước chọn cấu hình và ngưỡng có sử dụng nhãn validation; mô hình không được fit lại trên train + validation.

Mỗi mô hình được đánh giá theo hai thiết lập:

- **`default`**: detector ánh xạ dự đoán `-1` thành bot; Gradient Boosting dùng quyết định mặc định của bộ phân loại.
- **`validation_f1`**: áp dụng ngưỡng đã chọn trên validation, giữ nguyên khi đánh giá test.

Điểm dùng cho AP và ROC-AUC là `-decision_function` đối với detector và xác suất lớp bot đối với Gradient Boosting. Điểm càng lớn càng được mô hình xem là giống bot. Các chỉ số khác gồm precision, recall, F1 lớp bot, Macro-F1, accuracy và các ô TN/FP/FN/TP.

## 4. Kết quả trong notebook hiện tại

Các số liệu dưới đây lấy từ **output đã lưu trong notebook 04**, làm tròn đến bốn chữ số thập phân.

| Mô hình | Thiết lập | F1 bot trên test | AP trên test | ROC-AUC trên test |
| --- | --- | --- | --- | --- |
| Isolation Forest | `default` | 0,1365 | 0,2809 | 0,3685 |
| Isolation Forest | `validation_f1` | 0,4978 | 0,2809 | 0,3685 |
| One-Class SVM | `default` | 0,0944 | 0,3731 | 0,5643 |
| One-Class SVM | `validation_f1` | 0,4960 | 0,3731 | 0,5643 |
| Gradient Boosting | `default` | 0,6889 | 0,7889 | 0,8613 |
| Gradient Boosting | `validation_f1` | **0,7051** | **0,7889** | **0,8613** |

Cấu hình được chọn là **One-Class SVM: `nu=0.05`, `gamma=0.05`** và **Gradient Boosting: 200 cây, độ sâu 4**. Ngưỡng Gradient Boosting chọn trên validation là khoảng **0,369156**; khi áp dụng trên test, precision bot đạt **0,6795**, recall bot **0,7328** và accuracy **0,7966**.

Gradient Boosting phân biệt hai lớp tốt nhất trong ba mô hình. Hai detector có F1 sau hiệu chỉnh gần mốc dự đoán tất cả tài khoản là bot (**0,498** trên test), cho thấy giả định bot là điểm bất thường chưa phù hợp với bốn đặc trưng hồ sơ trong dữ liệu này.

Notebook 04 phân tích Gradient Boosting bằng impurity importance trên train và permutation importance trên validation, với 10 lần xáo trộn:

| Đặc trưng | Impurity importance | Mức giảm AP validation khi xáo trộn |
| --- | --- | --- |
| `posting_frequency` | 0,3908 | 0,2082 |
| `account_age_days` | 0,2327 | 0,1533 |
| `follower_following_ratio` | 0,2208 | 0,1527 |
| `bio_length` | 0,1557 | 0,0773 |

`posting_frequency` có mức quan trọng lớn nhất theo cả hai cách đo. Feature importance phản ánh cách mô hình sử dụng đặc trưng, không chứng minh quan hệ nhân quả.

## 5. Cấu trúc dự án và đầu ra

```text
.
├── README.md
├── requirements.txt
├── dataset/
│   ├── twitter_human_bots_dataset.csv
│   └── processed/                # features.csv và split.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_anomaly_detection.ipynb
│   ├── 03_supervised_baseline.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py                 # Đường dẫn, nhãn, seed và đặc trưng
│   ├── data.py                   # Đọc dữ liệu, tạo đặc trưng, chia tập
│   ├── models.py                 # Huấn luyện, chọn cấu hình/ngưỡng, lưu/tải
│   └── evaluate.py               # Chỉ số và feature importance
├── tests/
│   └── test_pipeline.py
├── models/                       # Mô hình từ notebook 02–03
└── reports/                      # Bảng kết quả và thư mục figures/
```

Các file sau được tạo hoặc ghi lại khi chạy notebook:

| Bước | Đầu ra |
| --- | --- |
| Notebook 01 | `dataset/processed/features.csv`, `split.csv`; biểu đồ phân phối và tương quan trên train |
| Notebook 02 | `models/isolation_forest.joblib`, `models/one_class_svm.joblib`; `reports/anomaly_validation_metrics.csv`; biểu đồ điểm bất thường |
| Notebook 03 | `models/gradient_boosting.joblib`; `reports/supervised_validation_search.csv`, `supervised_validation_metrics.csv`; biểu đồ Precision–Recall và ma trận nhầm lẫn validation |
| Notebook 04 | `reports/metrics.csv`, `comparison_validation_metrics.csv`, `feature_importance.csv`; biểu đồ Precision–Recall test, ma trận nhầm lẫn và feature importance |

Biểu đồ được lưu trong `reports/figures/`. Mỗi file mô hình chứa pipeline đã fit, cấu hình, ngưỡng và danh sách đặc trưng. Nếu bản source đã được dọn kết quả, chạy lại bốn notebook để tạo đầy đủ các file trên.

## 6. Cài đặt và chạy

Môi trường mục tiêu là **Python 3.14.x, 64-bit**, với thư viện trong `requirements.txt`. Giải nén source và mở terminal tại thư mục chứa `README.md`.

### Windows — PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

### Linux/macOS — Bash

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

Trong JupyterLab, mở từng notebook trong `notebooks/` và chọn **Run All** theo thứ tự **01 → 02 → 03 → 04**. Thư mục làm việc của notebook cần là thư mục gốc dự án hoặc `notebooks/` để tìm được `src/`.

Nếu dùng VS Code, mở thư mục dự án, cài extension Python và Jupyter, rồi chọn kernel `.venv` trong mỗi notebook trước khi chạy. Notebook 04 cần ba file mô hình được tạo bởi notebook 02 và 03.

Để chỉ tạo bảng đặc trưng bằng PowerShell:

```powershell
.\.venv\Scripts\python.exe -m src.data
```

Lệnh này tạo `features.csv`; việc chia tập và tạo báo cáo được thực hiện trong các notebook.

Chạy bộ kiểm thử bằng PowerShell:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Trên Bash sau khi kích hoạt `.venv`, dùng `python -m src.data` và `python -B -m unittest discover -s tests -v`. Bộ kiểm thử gồm 12 test về tạo đặc trưng, dữ liệu không hợp lệ, phân hoạch, điểm mô hình, chọn ngưỡng và chỉ số đánh giá.

## 7. Giới hạn và tái lập kết quả

- Chỉ sử dụng bốn đặc trưng hồ sơ, chưa xét nội dung bài đăng hoặc mạng lưới tương tác.
- Tuổi tài khoản lấy từ dữ liệu nguồn; tần suất đăng là trung bình cả vòng đời, không phản ánh riêng hoạt động gần đây.
- Kết quả dựa trên một bộ dữ liệu và một lần chia tập. Test đã được quan sát trong các thí nghiệm trước, nên không phải tập kiểm thử hoàn toàn chưa từng được xem.
- Không chọn lại cấu hình hoặc ngưỡng dựa trên kết quả test. Các chỉ số validation sau chọn cấu hình/ngưỡng không phải ước lượng độc lập trên dữ liệu mới.
- Giữ dữ liệu nguồn và `split.csv`. Nếu thay dữ liệu hoặc mã dùng chung, chạy lại toàn bộ 01–04; nếu thay bước huấn luyện, chạy lại notebook tương ứng và notebook 04.
- `requirements.txt` dùng khoảng phiên bản, chưa khóa chính xác toàn bộ môi trường; kết quả có thể chênh lệch giữa các phiên bản thư viện.
- Nhãn bot không đủ để kết luận một tài khoản giả mạo danh tính hoặc lừa đảo.

## Thành viên

- Nguyễn Bá Thọ
- Lê Hải Sơn
