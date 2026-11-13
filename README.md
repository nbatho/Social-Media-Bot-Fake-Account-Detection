# Social Media Bot & Fake Account Detection

Dự án học máy phát hiện tài khoản Twitter có hành vi giống bot thông qua các đặc trưng hồ sơ. Dự án sử dụng Isolation Forest và One-Class SVM để phát hiện bất thường, đối chiếu với Gradient Boosting và phân tích các đặc trưng ảnh hưởng đến kết quả phân loại.

> **Phạm vi theo đề bài:** trích xuất bốn đặc trưng hồ sơ, huấn luyện Isolation Forest và One-Class SVM, đối chiếu với Gradient Boosting và xác định các đặc trưng quan trọng. Notebook 01–04 trình bày lần lượt dữ liệu, anomaly detection, supervised benchmark và so sánh/feature importance. Tập test đã được quan sát trong các thí nghiệm trước; lịch sử sử dụng được ghi rõ trong notebook 04.

## Mục tiêu

- Trích xuất tỷ lệ follower/following, tần suất đăng bài, tuổi tài khoản và độ dài bio.
- Áp dụng **Isolation Forest** và **One-Class SVM** để phát hiện hành vi bất thường.
- Xây dựng mô hình **Gradient Boosting Classifier** trên nhãn `bot`/`human` để so sánh.
- Đánh giá các mô hình trên cùng tập kiểm thử.
- Xác định các đặc trưng quan trọng ảnh hưởng đến phân loại bot.

| Yêu cầu đề bài | Phần thực hiện |
| --- | --- |
| Follower/following ratio, posting frequency, account age, bio length | `src/data.py`, notebook 01 |
| Isolation Forest và One-Class SVM | `src/models.py`, notebook 02 |
| Benchmark với Gradient Boosting | `src/models.py`, notebook 03–04 |
| Top feature importances | Notebook 04, `reports/feature_importance.csv` |

Phạm vi phân loại dựa trên nhãn `bot` và `human` của dữ liệu. Nhãn bot hoặc điểm bất thường không đủ để kết luận tài khoản giả mạo danh tính hay có hành vi lừa đảo.

## Dataset

Nguồn: [Kaggle — Twitter Bots Accounts](https://www.kaggle.com/datasets/davidmartngutirrez/twitter-bots-accounts), tác giả David Martín Gutiérrez.

File hiện có: [`dataset/twitter_human_bots_dataset.csv`](dataset/twitter_human_bots_dataset.csv).

| Thuộc tính | Giá trị |
| --- | --- |
| Số dòng dữ liệu | 37.438 |
| Số cột nguồn | 20, gồm một cột index không tên |
| Nhãn `bot` | 12.425 (33,19%) |
| Nhãn `human` | 25.013 (66,81%) |
| ID trùng | 0 |
| ID hoặc nhãn rỗng | 0 |

Thống kê được kiểm tra ngày **25/10/2026** trên CSV hồ sơ đầy đủ đang có trong repository. Các cột cần dùng: `id`, `account_type`, `followers_count`, `friends_count`, `statuses_count`, `description`, `account_age_days`. `friends_count` là số tài khoản đang theo dõi.

Có **7.256 bio là ô rỗng** và **một bio chứa chuỗi `"None"`**. Pipeline giữ nguyên văn bản `"None"` (độ dài 4), vì chưa có bằng chứng đây là ký hiệu thiếu của nguồn. Không dùng các token thiếu mặc định của pandas cho cột văn bản hoặc ID.

## Đặc trưng và quy tắc xử lý

| Đặc trưng | Công thức / ý nghĩa |
| --- | --- |
| `follower_following_ratio` | `followers_count / max(friends_count, 1)` |
| `posting_frequency` | `statuses_count / max(account_age_days, 1)`; số bài/ngày trung bình trong vòng đời tài khoản |
| `account_age_days` | Tuổi tài khoản do nguồn cung cấp; số âm/không đọc được chuyển thành thiếu |
| `bio_length` | Độ dài chuỗi `description`; ô CSV rỗng có độ dài 0 theo giả định của nhóm |

Tuổi tài khoản là giá trị lịch sử của nguồn, **không phải tuổi tại ngày chạy chương trình**. CSV không có `observed_at`, nên không xác định được chính xác thời điểm quan sát; pipeline dùng trực tiếp `account_age_days`, không tính lại từ `created_at` hoặc ngày hiện tại. Tần suất đăng là trung bình toàn bộ vòng đời, không phản ánh riêng hoạt động gần đây.

Quy tắc chất lượng dữ liệu:

- Đọc ID dưới dạng chuỗi, ánh xạ `id` → `user_id`, `human` → `label=0`, `bot` → `label=1`. Giữ tên nguồn `friends_count` cho số tài khoản đang theo dõi.
- Thiếu cột bắt buộc, ID rỗng/trùng hoặc nhãn ngoài `human`/`bot` sẽ báo lỗi và dừng; không âm thầm xóa dòng. Các cột khác của nguồn không tham gia mô hình.
- Bốn cột số được đọc bằng `pd.to_numeric(errors="coerce")`; số không đọc được và số âm trở thành giá trị thiếu. Bản rút gọn không phân loại chi tiết lỗi số hay kiểm tra số đếm nguyên.
- Following bằng 0 hoặc tuổi bằng 0 là hợp lệ: mẫu số dùng 1. Mẫu số thiếu hoặc tuổi âm vẫn cho đặc trưng thiếu. Median imputer và scaler chỉ được fit trong pipeline mô hình trên train.
- Bio rỗng có độ dài 0 vì dữ liệu không phân biệt được bio thực sự rỗng với bio chưa thu thập được. `keep_default_na=False` giữ nguyên văn bản `None`, `NA`, `nan`. Unicode được đếm theo code point của Python; khoảng trắng và xuống dòng được tính vào độ dài, không cắt nội dung bio.
- Tự tính `posting_frequency` từ `statuses_count` và `account_age_days`; không dùng cột tần suất đã làm tròn của nguồn.

Đầu ra:

- `dataset/processed/features.csv`: `user_id, follower_following_ratio, posting_frequency, account_age_days, bio_length, label`, đúng thứ tự này. Giá trị số thiếu được ghi thành ô CSV rỗng.
- Notebook 01 hiển thị số giá trị thiếu, phân bố nhãn và quy mô các tập; không sinh báo cáo chất lượng JSON riêng.
- Chỉ bốn cột trong `FEATURE_COLUMNS` là đầu vào mô hình; `user_id` và `label` không phải đặc trưng dự đoán.

## Phương pháp

| Mô hình | Vai trò |
| --- | --- |
| Isolation Forest | Phát hiện các mẫu dễ bị cô lập trong không gian đặc trưng |
| One-Class SVM | Xác định các mẫu nằm ngoài vùng dữ liệu được mô hình học |
| Gradient Boosting Classifier | Phân loại có giám sát và phân tích feature importance |

Các mô hình dùng cùng bộ đặc trưng và tập kiểm thử. Tiền xử lý được fit trên train; cấu hình và ngưỡng được chọn trên validation. Với các mô hình phát hiện bất thường, cần phân biệt kết quả ở ngưỡng mặc định và kết quả có sử dụng nhãn validation để hiệu chỉnh.

## Đánh giá và giải thích

Quy ước **`human = 0`, `bot = 1`**. Các chỉ số gồm Precision, Recall, F1 cho lớp bot, Macro-F1, Average Precision (AP), ROC-AUC và Accuracy; kèm các ô TN/FP/FN/TP và đường Precision–Recall. Các tập validation/test hiện có đủ hai lớp để tính AP và ROC-AUC.

Notebook 04 tính impurity importance trên train và permutation importance trên validation để xác định các đặc trưng quan trọng. Kết quả nằm trong `reports/feature_importance.csv` và các biểu đồ. `reports/metrics.csv` so sánh ba mô hình trên test; `reports/comparison_validation_metrics.csv` đối chiếu validation. Không xem các chỉ số sau chọn cấu hình/ngưỡng trên cùng validation là ước lượng độc lập cho dữ liệu mới.

## Công nghệ dự kiến

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn và Jupyter Notebook.

## Cấu trúc dự án

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── dataset/
│   ├── twitter_human_bots_dataset.csv  # CSV nguồn, giữ nguyên
│   └── processed/                     # Features và split sinh ra
├── notebooks/
│   ├── 00_full_pipeline_colab.ipynb    # Bản gộp độc lập để tải lên Colab
│   ├── 01_data_exploration.ipynb       # Kiểm tra dữ liệu và EDA
│   ├── 02_anomaly_detection.ipynb      # Isolation Forest, One-Class SVM
│   ├── 03_supervised_baseline.ipynb    # Gradient Boosting benchmark
│   └── 04_evaluation.ipynb            # So sánh và giải thích mô hình
├── src/
│   ├── __init__.py
│   ├── config.py                      # 22 dòng: đường dẫn, seed, nhãn, đặc trưng
│   ├── data.py                        # 67 dòng: đọc CSV, tạo đặc trưng, chia tập
│   ├── models.py                      # 107 dòng: train, tìm cấu hình/ngưỡng, lưu/tải
│   └── evaluate.py                    # 51 dòng: chỉ số, so sánh, feature importance
├── tests/
│   └── test_pipeline.py               # 12 kiểm thử chính
├── models/                            # Pipeline đã fit; sinh lại từ code, được Git bỏ qua
└── reports/
    ├── metrics.csv                    # So sánh 3 mô hình × 2 thiết lập trên test
    ├── feature_importance.csv         # Impurity và permutation importance
    ├── *_validation_*.csv             # Kết quả và tìm cấu hình trên validation
    ├── history/                       # Kết quả model tạm trước đây
    └── figures/                       # Biểu đồ từ notebook 01–04
```

Notebook là nơi thử nghiệm và trình bày kết quả; các hàm dùng chung nằm trong `src/`. Đường dẫn trong `src/config.py` được tính từ vị trí dự án, không gắn với máy của một thành viên.

**Phần hiện chạy được:** bốn mô-đun chính có tổng **247 dòng** (ngoài `__init__.py`). `python -m src.data` đọc CSV, tạo và lưu `features.csv`; `load_profiles()` trả về một DataFrame, `build_features(profiles)` tạo bốn đặc trưng, `run_pipeline()` trả về bảng đặc trưng và `split_features(features)` trả `(train, validation, test)`. Ba hàm huấn luyện trong `src.models` nhận cả bảng và tự lấy `FEATURE_COLUMNS`, nên ID/label không lọt vào đầu vào. `bot_scores()` trả điểm càng lớn càng giống bot cho cả ba mô hình; `default_predictions()` dùng quyết định mặc định, `select_threshold()` trả ngưỡng F1 validation. `search()` dùng chung để chọn cấu hình SVM và Gradient Boosting theo AP validation; `save_model()`/`load_model()` lưu/tải pipeline, cấu hình, ngưỡng và danh sách đặc trưng. `src.evaluate` tính chỉ số, so sánh model đã lưu và tính hai loại feature importance.

**Supervised (notebook 03, chỉ trên validation):** Gradient Boosting được chọn theo AP từ lưới cố định 6 cấu hình: 100/200 cây × độ sâu 2/3/4, learning rate 0,1, min_samples_leaf=5, subsample=1, seed=42. Cấu hình chọn 200 cây, độ sâu 4 đạt AP **0,7876**, ROC-AUC **0,8615**. F1 bot ở ngưỡng 0,5 là **0,6953**; ngưỡng chọn theo F1 validation **0,369156** đạt Precision **0,6851**, Recall **0,7336**, F1 bot **0,7085**. Median imputer chỉ fit trên train, không refit trên train+validation. Notebook 04 tải model bằng `load_model("gradient_boosting")` và dùng ngưỡng `bundle["threshold"]`. Con số test 0,863 trong ghi chú cũ thuộc model tạm 300 cây/max_depth=5, giữ trong `reports/history/2026-11-09_interim_metrics.csv`; model chính thức hiện dùng cấu hình 200 cây/max_depth=4.

**Phát hiện bất thường (notebook 02, chỉ trên validation):** Isolation Forest đạt ROC-AUC 0,378 và One-Class SVM (`nu=0.05`, `gamma=0.05`) đạt 0,565, so với 0,5 của xếp hạng ngẫu nhiên. Ngưỡng chọn trên validation chỉ cho F1 lớp bot khoảng 0,50, ngang mốc gắn nhãn tất cả là bot (0,498). Bot trong dữ liệu có hồ sơ điển hình, còn điểm ngoại lai chủ yếu là tài khoản người thật có nhiều follower. Notebook lưu model vào `models/*.joblib` (kèm cấu hình, đặc trưng và ngưỡng), bảng `reports/anomaly_validation_metrics.csv` và biểu đồ phân phối điểm trong `reports/figures/`. Kết quả test được báo cáo ở notebook 04; không đổi cấu hình/ngưỡng sau khi xem test.

**Chia tập:** nguồn không có split chính thức, nên tài khoản được chia 60/20/20, stratify theo nhãn, seed `RANDOM_STATE`. Mỗi `user_id` thuộc đúng một tập. Lần chạy đầu lưu `user_id → split` vào `dataset/processed/split.csv`; các lần sau dùng lại file này để mọi notebook có cùng tập kiểm thử. Nếu file không khớp đúng tập `user_id` hiện có, hàm báo lỗi chứ không tự tạo split mới. Trên CSV hiện tại: train 22.462, validation 7.488, test 7.488 dòng; tỷ lệ bot mỗi tập là 33,19%.

`dataset/processed/` và `models/` được giữ trong Git bằng `.gitkeep`; dữ liệu và mô hình sinh ra bên trong được bỏ qua. Biểu đồ và bảng kết quả được lưu trong `reports/` để nhóm chọn đưa vào báo cáo/repository.

## Cài đặt và chạy

### 1. Chuẩn bị

- **Python 3.14.x bản 64-bit**; môi trường mục tiêu của nhóm là **Python 3.14.5**. Có thể tải Python từ [trang chính thức](https://www.python.org/downloads/windows/).
- **Git** để tải và quản lý mã nguồn.
- **VS Code** với extension **Python** và **Jupyter** của Microsoft nếu muốn chạy notebook trong trình soạn thảo.

Hướng dẫn hỗ trợ **PowerShell trên Windows** và **Bash** (Git Bash trên Windows hoặc Bash trên Linux/macOS/WSL). Chạy lệnh tại thư mục gốc repository, nơi chứa `README.md`, `requirements.txt` và thư mục `dataset/`; chọn phần tương ứng với terminal và hệ điều hành của bạn.

### 2. Tạo môi trường ảo `.venv`

**PowerShell — Windows:**

```powershell
python --version
python -m venv .venv
```

Các bước PowerShell bên dưới gọi trực tiếp Python trong `.venv`, vì vậy không cần chạy `Activate.ps1`.

**Bash — Git Bash trên Windows:**

```bash
python --version
python -m venv .venv
source .venv/Scripts/activate
```

**Bash — Linux/macOS/WSL:**

```bash
python3.14 --version
python3.14 -m venv .venv
source .venv/bin/activate
```

Kết quả kiểm tra phiên bản mong đợi là `Python 3.14.x`. Trên Windows có nhiều phiên bản Python, có thể thay lệnh tạo môi trường bằng `py -3.14 -m venv .venv`. Nếu lệnh `python` mở Microsoft Store hoặc không chạy được, kiểm tra lại Python/PATH hoặc dùng lệnh `py` tương ứng. Trên Linux/macOS/WSL, cần cài Python 3.14 và thành phần `venv` tương ứng trước khi tạo môi trường.

Chỉ tạo `.venv` một lần cho mỗi môi trường hệ điều hành. Với Bash, sau khi `source` thành công, các lệnh `python` bên dưới dùng Python trong `.venv`. Mỗi lần mở terminal Bash mới, chạy lại lệnh `source` đúng với hệ điều hành; dùng `deactivate` khi muốn thoát môi trường. Nếu đổi từ Windows sang WSL/Linux/macOS, tạo `.venv` riêng trong bản checkout của hệ điều hành đó, không dùng chung thư mục `.venv` của Windows.

### 3. Cài thư viện

**PowerShell:**

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Bash — sau khi kích hoạt `.venv`:**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

File [`requirements.txt`](requirements.txt) gồm:

| Thư viện | Công dụng |
| --- | --- |
| pandas, NumPy | Đọc dữ liệu, làm sạch và tính đặc trưng |
| scikit-learn | Tiền xử lý, huấn luyện và đánh giá mô hình |
| Matplotlib, Seaborn | Trực quan hóa dữ liệu và kết quả |
| ipykernel | Chạy Python trong notebook |
| JupyterLab | Mở và chạy notebook trên trình duyệt |
| joblib | Lưu và tải mô hình |

### 4. Kiểm tra môi trường

**PowerShell:**

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import pandas, numpy, sklearn, matplotlib, seaborn, ipykernel, jupyterlab, joblib; print('Môi trường OK')"
.\.venv\Scripts\python.exe -m pip check
```

**Bash — sau khi kích hoạt `.venv`:**

```bash
python --version
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, ipykernel, jupyterlab, joblib; print('Môi trường OK')"
python -m pip check
```

Nếu các thư viện import thành công, chương trình in `Môi trường OK`. Lệnh `pip check` cần trả về `No broken requirements found`.

Sau khi cài requirements, chạy pipeline từ thư mục gốc dự án:

**PowerShell:**

```powershell
.\.venv\Scripts\python.exe -m src.data
```

**Bash — sau khi kích hoạt `.venv`:**

```bash
python -m src.data
```

Lệnh tạo `dataset/processed/features.csv`. Kết quả trên CSV hiện tại: **37.438 dòng × 6 cột**, giữ toàn bộ dòng, không có đặc trưng thiếu. Chạy lại ghi đè bảng đặc trưng, không ghi vào CSV nguồn. Notebook 01 tạo hoặc dùng lại split và hiển thị thống kê chính.

Chạy kiểm thử từ thư mục gốc:

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Bash (sau khi kích hoạt `.venv`):

```bash
python -m unittest discover -s tests -v
```

Bộ kiểm thử dùng `unittest` có sẵn, gồm **12 test** trong `tests/test_pipeline.py`: công thức và mẫu số 0, bio rỗng/chuỗi `None`, số âm/chữ thành thiếu, ID/nhãn/cột lỗi, chia tập đầy đủ và dùng lại, dữ liệu không khớp split, chiều điểm detector, nhãn không lọt vào đặc trưng, Gradient Boosting học tín hiệu, ngưỡng tối đa F1 và các chỉ số tính tay.

Tái tạo model và báo cáo bằng Run All notebook 01 → 02 → 03 → 04. Notebook 02 lưu hai detector; notebook 03 lưu `models/gradient_boosting.joblib`, bảng validation và bảng tìm kiếm. Notebook 04 tải các model đã lưu để đánh giá, không huấn luyện lại.

### 5. Chạy notebook

**Google Colab — một notebook duy nhất:**

1. Tải [`notebooks/00_full_pipeline_colab.ipynb`](notebooks/00_full_pipeline_colab.ipynb) lên Colab.
2. Chọn runtime Python / CPU và **Run all**. Khi được yêu cầu, tải lên `dataset/twitter_human_bots_dataset.csv` từ máy.
3. Notebook tự cài thư viện, định nghĩa các hàm trực tiếp trong cell (không tạo thư mục `src/`), chạy đủ bốn phần theo thứ tự và tải ZIP kết quả ở cell cuối. Không cần tải riêng `src/` hoặc bốn notebook gốc. Lưu ZIP trước khi đóng runtime; ZIP chứa cả `split.csv` để dùng lại phân hoạch.

Bản Colab dùng cùng công thức đặc trưng, seed, lưới cấu hình và cách chọn ngưỡng với notebook 01–04. Khi chạy local, kết quả nằm trong thư mục riêng `bot_detection_colab/`; nếu tìm thấy CSV và split của dự án, notebook sao chép chúng để dùng lại. Sau khi sửa `src/` hoặc notebook gốc, chạy `python scripts/build_colab_notebook.py` để tạo lại bản gộp.

**Trong VS Code:**

1. Mở thư mục gốc dự án bằng **File → Open Folder**.
2. Mở Command Palette (`Ctrl + Shift + P` trên Windows/Linux hoặc `Cmd + Shift + P` trên macOS), chọn **Python: Select Interpreter** rồi chọn Python của dự án: `.venv\Scripts\python.exe` trên Windows hoặc `.venv/bin/python` trên Linux/macOS/WSL.
3. Chọn **Select Kernel → Python Environments → .venv** trong từng notebook.
4. Chạy **Run All** lần lượt notebook **01 → 02 → 03 → 04**. Notebook 01 kiểm tra dữ liệu và EDA; 02 tạo model anomaly; 03 tạo model supervised; 04 chỉ tải model, so sánh validation rồi đánh giá test và tính importance. Không đổi cấu hình/ngưỡng dựa trên kết quả test.

**Hoặc dùng JupyterLab trên trình duyệt:**

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

Bash, sau khi kích hoạt `.venv`:

```bash
python -m jupyterlab
```

Mở địa chỉ local được in trong terminal nếu trình duyệt chưa tự mở. Vào `notebooks/` và mở notebook 01 với kernel Python của môi trường; nhấn `Ctrl + C` trong terminal khi muốn dừng JupyterLab.

Trong notebook 01, chạy cell thiết lập `project_root`, sau đó phần **Pipeline đặc trưng dùng chung**. Phần này gọi các hàm trong `src/` để đọc/kiểm tra dữ liệu, tính đặc trưng, lưu đầu ra và hiển thị số giá trị thiếu. Các cell thực hành trước đó được giữ nguyên để tham khảo; cách đọc chính thức là `load_profiles()` để bảo toàn văn bản bio. Có thể chạy lại pipeline từ notebook mà không chép công thức hoặc logic làm sạch vào cell.

`compare_models()` trong `src.evaluate` đánh giá các model cố định, không huấn luyện lại; notebook 04 tạo nhận xét từ số liệu hiện tại. Các kết quả model tạm trước đó được giữ trong `reports/history/`, không thuộc bảng so sánh hiện tại. `docs/` tiếp tục được Git bỏ qua.

**Đã kiểm chứng ngày 12/11/2026:** 12/12 test đạt; notebook 01–04 chạy đủ mọi cell và lưu đầy đủ biểu đồ. Bảng đặc trưng 37.438 dòng, các chỉ số giữ lại trong `metrics.csv` và cả hai loại feature importance khớp bản trước khi rút gọn, sai lệch số bằng 0. CSV nguồn và `split.csv` giữ nguyên; SVM vẫn chọn `nu=0.05, gamma=0.05`, Gradient Boosting chọn 200 cây, độ sâu 4.

### Quy tắc tái lập kết quả

Bản rút gọn không còn tự kiểm tra SHA-256 nguồn/split/code, `run_id` hoặc phiên bản scikit-learn để chặn model cũ. `load_model()` chỉ đối chiếu danh sách đặc trưng. Nhóm phải tuân thủ:

1. **Chạy theo thứ tự 01 → 02 → 03 → 04.** Mỗi lần sửa notebook 02 hoặc 03 phải chạy lại notebook 04; nếu sửa dữ liệu hoặc code dùng chung, chạy lại cả bốn.
2. **Không xóa `dataset/processed/split.csv`.** Đây là phân hoạch cố định dùng chung. Nếu tạo lại split, không mặc định các kết quả cũ vẫn so sánh được; phải huấn luyện và đánh giá lại toàn bộ.
3. **Không sửa lưới cấu hình hay cách chọn ngưỡng sau khi đã xem test.** Nếu buộc phải sửa, ghi rõ trong báo cáo rằng test đã được xem trước. Lịch sử quan sát test hiện có được giữ trong notebook 04.

### 6. Đồng bộ môi trường giữa các thành viên

Mỗi thành viên tạo `.venv` trên máy của mình và cài từ cùng `requirements.txt`. File này quy định khoảng phiên bản thư viện, chưa cố định toàn bộ phiên bản chính xác. Khi nhóm đã kiểm tra môi trường chạy ổn, có thể lưu một bản cố định:

**PowerShell:**

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements-lock.txt
```

**Bash — sau khi kích hoạt `.venv`:**

```bash
python -m pip freeze > requirements-lock.txt
```

Máy còn lại dùng cùng Python 3.14.x, tạo `.venv` và cài bằng:

**PowerShell:**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

**Bash — sau khi kích hoạt `.venv`:**

```bash
python -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` chỉ có sau khi chạy lệnh xuất môi trường. Các file requirements được chia sẻ qua Git; `.venv/` và cache được bỏ qua bởi `.gitignore`.

## Thành viên

- **Nguyễn Bá Thọ**
- **Lê Hải Sơn**

## Tài liệu tham khảo

- [scikit-learn — Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [scikit-learn — GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [scikit-learn — Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
