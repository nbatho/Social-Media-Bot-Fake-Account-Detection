# Social Media Bot & Fake Account Detection

Dự án học máy phát hiện tài khoản Twitter có hành vi giống bot thông qua các đặc trưng hồ sơ. Dự án sử dụng Isolation Forest và One-Class SVM để phát hiện bất thường, đối chiếu với Gradient Boosting và phân tích các đặc trưng ảnh hưởng đến kết quả phân loại.

> **Trạng thái:** đã có pipeline đặc trưng, split dùng chung, EDA trên train, hai mô hình anomaly (notebook 02), Dummy và Gradient Boosting (notebook 03), cùng hàm đánh giá có kiểm thử. Kết quả hiện báo cáo trên validation; notebook 04 vẫn là khung. Ghi chú cũ nhắc test đã được dùng và AUC 0,863 chưa có thí nghiệm nguồn để xác minh; lần triển khai supervised này không đánh giá hoặc chọn cấu hình/ngưỡng bằng test.

## Mục tiêu

- Trích xuất tỷ lệ follower/following, tần suất đăng bài, tuổi tài khoản và độ dài bio.
- Áp dụng **Isolation Forest** và **One-Class SVM** để phát hiện hành vi bất thường.
- Xây dựng mô hình **Gradient Boosting Classifier** trên nhãn `bot`/`human` để so sánh.
- Đánh giá các mô hình trên cùng tập kiểm thử.
- Xác định các đặc trưng quan trọng và phân tích những trường hợp dự đoán sai.

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

Thống kê được kiểm tra ngày **25/10/2026** trên CSV hồ sơ đầy đủ đang có trong repository. Các cột cần dùng: `id`, `account_type`, `followers_count`, `friends_count`, `statuses_count`, `created_at`, `description`, `account_age_days`, `average_tweets_per_day`. `friends_count` là số tài khoản đang theo dõi.

Có **7.256 bio là ô rỗng** và **một bio chứa chuỗi `"None"`**. Pipeline giữ nguyên văn bản `"None"` (độ dài 4), vì chưa có bằng chứng đây là ký hiệu thiếu của nguồn. Không dùng các token thiếu mặc định của pandas cho cột văn bản hoặc ID.

## Đặc trưng và quy tắc xử lý

| Đặc trưng | Công thức / ý nghĩa |
| --- | --- |
| `follower_following_ratio` | `followers_count / max(following_count, 1)` |
| `posting_frequency` | `statuses_count / max(account_age_days, 1)`; số bài/ngày trung bình trong vòng đời tài khoản |
| `account_age_days` | Tuổi tài khoản do nguồn cung cấp, đã kiểm tra kiểu số, giá trị thiếu, âm và vô cực |
| `bio_length` | Độ dài chuỗi `description`; bio rỗng/null có độ dài 0 theo giả định của nhóm |

Tuổi tài khoản là giá trị lịch sử của nguồn, **không phải tuổi tại ngày chạy chương trình**. CSV không có `observed_at`, nên không xác định được chính xác thời điểm quan sát; pipeline không tạo giả trường này, không lấy ngày hiện tại/ngày tải file và không tính lại tuổi từ `created_at`. `created_at` được kiểm tra khả năng phân tích ngày giờ, nhưng không dùng để tính đặc trưng. Tần suất đăng là trung bình toàn bộ vòng đời, không phản ánh riêng hoạt động gần đây.

Quy tắc chất lượng dữ liệu:

- Đọc ID dưới dạng chuỗi, ánh xạ `id` → `user_id`, `friends_count` → `following_count`, `human` → `label=0`, `bot` → `label=1`.
- Chỉ bỏ cột index không tên khi giá trị đúng bằng dãy `0..n-1`. Thiếu cột bắt buộc, ID rỗng/trùng, nhãn không hợp lệ hoặc nhãn mâu thuẫn sẽ báo lỗi và dừng; không âm thầm xóa dòng.
- Các trường số thiếu, không chuyển được sang số, âm hoặc vô cực được thống kê và chuyển thành giá trị thiếu. Các số đếm phải là số nguyên; tuổi tính theo ngày có thể là số thập phân không âm. Numeric dùng kiểu nullable `Float64` để giữ giá trị thiếu. Các token `""`, `NA`, `N/A`, `NaN`, `null`, `None` (không phân biệt hoa/thường, sau khi bỏ khoảng trắng hai đầu) chỉ được coi là thiếu trong trường số.
- Following bằng 0 hoặc tuổi bằng 0 là hợp lệ: mẫu số dùng 1. Mẫu số thiếu hoặc tuổi âm không được thay bằng 1; đặc trưng phụ thuộc vẫn thiếu. Không điền trung bình/median, không fit scaler và không loại dòng có đặc trưng thiếu. Nếu phép tính tràn số, kết quả được chuyển thành thiếu và ghi riêng trong báo cáo.
- Bio null/rỗng có độ dài 0 vì dữ liệu không phân biệt được bio thực sự rỗng với bio chưa thu thập được. Giữ nguyên văn bản `None`, `NA`, `nan`; không biến null thành chuỗi `"nan"`. Unicode được đếm theo code point của Python, không phải số ký tự hiển thị; khoảng trắng và xuống dòng được tính vào độ dài, không cắt nội dung bio.
- Tự tính `posting_frequency` rồi so sánh với `average_tweets_per_day`. Báo cáo giả định nguồn làm tròn đến 3 chữ số thập phân, dùng sai số tuyệt đối `0.0005` cộng biên số thực `1e-12`. Sai lệch lớn hơn được thống kê, không tự động coi là lỗi hoặc loại dòng.

Đầu ra:

- `dataset/processed/features.csv`: `user_id, follower_following_ratio, posting_frequency, account_age_days, bio_length, label`, đúng thứ tự này. Giá trị số thiếu được ghi thành ô CSV rỗng.
- `reports/feature_summary.json`: chất lượng từng cột, số dòng giữ/loại, công thức, giả định, so sánh tần suất và SHA-256 nguồn trước/sau.
- Chỉ bốn cột trong `FEATURE_COLUMNS` là đầu vào mô hình; `user_id` và `label` không phải đặc trưng dự đoán.

## Phương pháp

| Mô hình | Vai trò |
| --- | --- |
| Dummy Classifier | Baseline dự đoán lớp phổ biến nhất |
| Isolation Forest | Phát hiện các mẫu dễ bị cô lập trong không gian đặc trưng |
| One-Class SVM | Xác định các mẫu nằm ngoài vùng dữ liệu được mô hình học |
| Gradient Boosting Classifier | Phân loại có giám sát và phân tích feature importance |

Các mô hình dùng cùng bộ đặc trưng và tập kiểm thử. Tiền xử lý được fit trên train; cấu hình và ngưỡng được chọn trên validation. Với các mô hình phát hiện bất thường, cần phân biệt kết quả ở ngưỡng mặc định và kết quả có sử dụng nhãn validation để hiệu chỉnh.

## Đánh giá và giải thích

Quy ước **`human = 0`, `bot = 1`**. Các chỉ số gồm Precision, Recall, F1 cho lớp bot, Macro-F1, Average Precision (AP), ROC-AUC và Accuracy; kèm confusion matrix, đường Precision–Recall và thời gian chạy.

Feature importance của Gradient Boosting và permutation importance sẽ được bổ sung trong notebook 04. Kết quả supervised trên validation đã có trong `reports/supervised_validation_metrics.csv`; không xem các chỉ số sau chọn cấu hình/ngưỡng trên cùng validation là ước lượng độc lập cho dữ liệu mới.

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
│   ├── 01_data_exploration.ipynb       # Kiểm tra dữ liệu và EDA
│   ├── 02_anomaly_detection.ipynb      # Isolation Forest, One-Class SVM
│   ├── 03_supervised_baseline.ipynb    # Dummy, Gradient Boosting
│   └── 04_evaluation.ipynb            # So sánh và giải thích mô hình
├── src/
│   ├── __init__.py
│   ├── config.py                      # Đường dẫn, seed, nhãn, schema
│   ├── data.py                        # Đọc, kiểm tra dữ liệu, chia tập
│   ├── features.py                    # Tạo đặc trưng hồ sơ
│   ├── train.py                       # Các hàm huấn luyện dùng chung
│   ├── supervised.py                  # Tái tạo tìm kiếm, model và báo cáo notebook 03
│   └── evaluate.py                    # Các hàm tính chỉ số dùng chung
├── tests/                             # Kiểm thử dữ liệu và đặc trưng
├── models/                            # Pipeline đã fit; sinh lại từ code, được Git bỏ qua
└── reports/
    ├── feature_summary.json           # Chất lượng và giả định đặc trưng
    └── figures/                       # Dành cho biểu đồ giai đoạn sau
```

Notebook là nơi thử nghiệm và trình bày kết quả; các hàm dùng chung nằm trong `src/`. Đường dẫn trong `src/config.py` được tính từ vị trí dự án, không gắn với máy của một thành viên.

**Phần hiện chạy được:** `python -m src.data` audit CSV; `load_profiles()` trả về `(profiles, quality)` đã chuẩn hóa và kiểm tra; `build_features(profiles)` tính bốn đặc trưng; `python -m src.features` chạy toàn bộ pipeline và lưu kết quả; `split_features(features)` trả về `(train, validation, test)`. Notebook 01 gọi các hàm dùng chung và giữ lại các cell thực hành trước đó của người học. Bốn hàm `train_isolation_forest()`, `train_one_class_svm()`, `train_dummy()` và `train_gradient_boosting()` trả về pipeline đã fit. `anomaly_scores()`, `predict_bot()` và `select_threshold()` phục vụ anomaly; `bot_probabilities()` trả xác suất bot sau khi tìm đúng cột `classes_ == 1`. `evaluate_predictions()` tính chỉ số chung, trả AP/ROC-AUC là None nếu truth chỉ có một lớp; Macro-F1 luôn tính cả hai nhãn 0/1.

**Supervised (notebook 03, chỉ trên validation):** Dummy `most_frequent` dự đoán toàn human, Accuracy 0,6681, F1 bot 0 và ROC-AUC 0,5. Gradient Boosting được chọn theo AP từ lưới cố định 6 cấu hình: 100/200 cây × độ sâu 2/3/4, learning rate 0,1, min_samples_leaf=5, subsample=1, seed=42. Cấu hình chọn 200 cây, độ sâu 4 đạt AP **0,7876**, ROC-AUC **0,8615**. F1 bot ở ngưỡng 0,5 là **0,6953**; ngưỡng chọn theo F1 validation **0,369156** đạt Precision **0,6851**, Recall **0,7336**, F1 bot **0,7085**. Median imputer chỉ fit trên train, không refit trên train+validation. Xem [hướng dẫn bàn giao](reports/supervised_handoff.md) để tải model, áp dụng ngưỡng và đối chiếu provenance. Đây là kết quả validation mới có truy vết, không xác nhận con số 0,863 trong ghi chú cũ.

**Phát hiện bất thường (notebook 02, chỉ trên validation):** Isolation Forest đạt ROC-AUC 0,378 và One-Class SVM (`nu=0.05`, `gamma=0.05`) đạt 0,565, so với 0,5 của xếp hạng ngẫu nhiên. Ngưỡng chọn trên validation chỉ cho F1 lớp bot khoảng 0,50, ngang mốc gắn nhãn tất cả là bot (0,498). Bot trong dữ liệu có hồ sơ điển hình, còn điểm ngoại lai chủ yếu là tài khoản người thật có nhiều follower. Notebook lưu model vào `models/*.joblib` (kèm cấu hình, đặc trưng và ngưỡng), bảng `reports/anomaly_validation_metrics.csv` và biểu đồ phân phối điểm trong `reports/figures/`. Kết quả test sẽ được báo cáo ở notebook 04.

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
.\.venv\Scripts\python.exe -m src.features
```

**Bash — sau khi kích hoạt `.venv`:**

```bash
python -m src.features
```

Lệnh tạo `features.csv` và `feature_summary.json`. Kết quả trên CSV hiện tại: **37.438 dòng × 6 cột**, giữ toàn bộ dòng, không có đặc trưng thiếu hoặc vô cực. ID/label khớp nguồn theo thứ tự; nguồn được kiểm tra SHA-256 trước/sau. Chạy lại sẽ tạo lại hai đầu ra, không ghi vào CSV nguồn.

Chạy kiểm thử từ thư mục gốc:

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Bash (sau khi kích hoạt `.venv`):

```bash
python -m unittest discover -s tests -v
```

Bộ kiểm thử dùng `unittest` có sẵn: ID dài/giữ số 0 đầu, nhãn và ID lỗi, index nguồn, following/tuổi bằng 0, tuổi âm, dữ liệu số thiếu/không hợp lệ/vô cực, bio Unicode/null/xuống dòng, làm tròn, bảo toàn CSV nguồn và chạy lại nhất quán.

Bộ kiểm thử sau triển khai supervised: **36/36 test đạt**. Phần bổ sung kiểm tra fit median chỉ trên train, Dummy đúng lớp phổ biến nhất, xác suất đúng lớp bot kể cả khi thứ tự lớp đảo hoặc thiếu lớp, tính tái lập theo seed, pipeline giữ schema khi lưu/tải và lựa chọn cấu hình/ngưỡng không truy cập test.

Tái tạo Dummy/Gradient Boosting, bảng validation và metadata từ thư mục gốc:

```powershell
.\.venv\Scripts\python.exe -m src.supervised
```

Với Bash sau khi kích hoạt `.venv`: `python -m src.supervised`. Lệnh dùng lại `dataset/processed/split.csv`, chỉ fit trên train và chọn cấu hình/ngưỡng trên validation. Model nằm trong `models/dummy.joblib`, `models/gradient_boosting.joblib`; các báo cáo `supervised_validation_*.csv` và `supervised_run.json` nằm trong `reports/`. Mỗi lần chạy ghi đè bộ đầu ra này với `run_id` mới; lưu cùng model và báo cáo nếu cần giữ lịch sử thí nghiệm. Chạy notebook 03 để có thêm biểu đồ Precision–Recall/confusion matrix.

### 5. Chạy notebook

**Trong VS Code:**

1. Mở thư mục gốc dự án bằng **File → Open Folder**.
2. Mở Command Palette (`Ctrl + Shift + P` trên Windows/Linux hoặc `Cmd + Shift + P` trên macOS), chọn **Python: Select Interpreter** rồi chọn Python của dự án: `.venv\Scripts\python.exe` trên Windows hoặc `.venv/bin/python` trên Linux/macOS/WSL.
3. Mở [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) để kiểm tra dữ liệu và EDA trên train. Notebook 02 huấn luyện Isolation Forest và One-Class SVM; notebook 03 huấn luyện Dummy/Gradient Boosting, chọn cấu hình/ngưỡng và kiểm tra model sau lưu/tải. Notebook 04 vẫn là khung cho phần so sánh cuối.
4. Chọn **Select Kernel → Python Environments → .venv**, sau đó chạy từng cell.

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

Trong notebook 01, chạy cell thiết lập `project_root`, sau đó phần **Pipeline đặc trưng dùng chung**. Phần này gọi các hàm trong `src/` để đọc/kiểm tra dữ liệu, tính đặc trưng, lưu đầu ra và hiển thị thống kê chất lượng. Các cell thực hành trước đó được giữ nguyên để tham khảo; cách đọc chính thức là `load_profiles()` để bảo toàn văn bản bio. Có thể chạy lại pipeline từ notebook mà không chép công thức hoặc logic làm sạch vào cell.

Notebook 01–03 đã chạy lại được bằng môi trường dự án. Notebook 04 chạy các cell khung nhưng chưa thực hiện so sánh bốn mô hình, importance hoặc đánh giá cuối. Việc review của thành viên, xác minh lịch sử sử dụng test và tích hợp `main` còn chờ phần phối hợp với Sơn. `docs/` tiếp tục được bỏ qua bởi Git; hướng dẫn bàn giao có thể chia sẻ nằm trong `reports/supervised_handoff.md`.

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
