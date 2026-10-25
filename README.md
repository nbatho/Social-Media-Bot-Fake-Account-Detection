# Social Media Bot & Fake Account Detection

Dự án học máy phát hiện tài khoản Twitter có hành vi giống bot thông qua các đặc trưng hồ sơ. Dự án sử dụng Isolation Forest và One-Class SVM để phát hiện bất thường, đối chiếu với Gradient Boosting và phân tích các đặc trưng ảnh hưởng đến kết quả phân loại.

> **Trạng thái:** đã có khung mã nguồn, notebook và lệnh kiểm tra CSV. Đang chuẩn bị dữ liệu hồ sơ; chưa có mô hình được huấn luyện hoặc kết quả thực nghiệm.

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
| Các cột | `id`, `account_type` |
| Nhãn `bot` | 12.425 (33,19%) |
| Nhãn `human` | 25.013 (66,81%) |
| ID trùng | 0 |
| ID hoặc nhãn rỗng | 0 |

Thống kê được kiểm tra ngày **25/10/2026**. File hiện tại chỉ chứa ID và nhãn, chưa có đủ thông tin hồ sơ để tính các đặc trưng yêu cầu. Cần bổ sung metadata tương ứng hoặc sử dụng nguồn dữ liệu có hồ sơ và nhãn trước khi huấn luyện. Loại ID cần được xác minh trước khi ghép dữ liệu; ID không được sử dụng làm đặc trưng dự đoán.

## Đặc trưng dự kiến

| Đặc trưng | Mô tả |
| --- | --- |
| Follower/following ratio | Số người theo dõi chia cho số tài khoản đang theo dõi, có xử lý mẫu số bằng 0 |
| Posting frequency | Số bài đăng trung bình mỗi ngày trong vòng đời tài khoản tại thời điểm quan sát |
| Account age | Khoảng thời gian từ ngày tạo tài khoản đến thời điểm quan sát hồ sơ |
| Bio length | Số ký tự trong phần giới thiệu tài khoản |

Các đặc trưng này cần dữ liệu về số follower, following, tổng số bài đăng, ngày tạo tài khoản, bio và thời điểm quan sát. Hiện chúng chưa có trong CSV nguồn.

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

Feature importance của Gradient Boosting và permutation importance được sử dụng để phân tích ảnh hưởng của các đặc trưng. Kết quả sẽ được bổ sung sau khi hoàn thành thực nghiệm; hiện chưa có số liệu hiệu năng để công bố.

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
│   └── evaluate.py                    # Các hàm tính chỉ số dùng chung
├── models/                            # Pipeline/model đã fit
└── reports/
    └── figures/                       # Biểu đồ; bảng metrics ở reports/
```

Notebook là nơi thử nghiệm và trình bày kết quả; các hàm dùng chung nằm trong `src/`. Đường dẫn trong `src/config.py` được tính từ vị trí dự án, không gắn với máy của một thành viên.

**Phần hiện chạy được:** `python -m src.data` kiểm tra CSV nguồn; `load_profiles` đọc CSV hồ sơ đã chuẩn hóa và kiểm tra các cột bắt buộc. Các hàm chia tập, tạo đặc trưng, huấn luyện và đánh giá mới có khai báo cùng hướng dẫn TODO; gọi chúng sẽ báo `NotImplementedError` cho đến khi được triển khai. Các notebook có cell khởi tạo/import và danh sách công việc, chưa có kết quả mô hình.

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

Kiểm tra CSV nguồn bằng lệnh sau; lệnh này chỉ dùng thư viện chuẩn Python nên có thể chạy trước khi cài xong requirements:

**PowerShell:**

```powershell
.\.venv\Scripts\python.exe -m src.data
```

**Bash — sau khi kích hoạt `.venv`:**

```bash
python -m src.data
```

Lệnh in số dòng, các cột, phân bố nhãn, ID trùng, giá trị thiếu và các cột hồ sơ cần bổ sung. Với dữ liệu hiện tại, số dòng là `37438`, gồm `12425` bot và `25013` human.

### 5. Chạy notebook

**Trong VS Code:**

1. Mở thư mục gốc dự án bằng **File → Open Folder**.
2. Mở Command Palette (`Ctrl + Shift + P` trên Windows/Linux hoặc `Cmd + Shift + P` trên macOS), chọn **Python: Select Interpreter** rồi chọn Python của dự án: `.venv\Scripts\python.exe` trên Windows hoặc `.venv/bin/python` trên Linux/macOS/WSL.
3. Mở [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) để bắt đầu kiểm tra dữ liệu; tiếp tục notebook 02–04 khi có dữ liệu hồ sơ và đã triển khai các hàm tương ứng.
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

Trong notebook lưu tại thư mục gốc, chạy cell sau để kiểm tra dữ liệu:

```python
from pathlib import Path
import sys
import pandas as pd

print("Python đang dùng:", sys.executable)
data_path = Path("dataset/twitter_human_bots_dataset.csv")
df = pd.read_csv(data_path, dtype={"id": "string"})
print("Kích thước:", df.shape)
print("Các cột:", df.columns.tolist())
print(df["account_type"].value_counts(dropna=False))
df.head()
```

Với CSV hiện tại, kích thước mong đợi là `(37438, 2)`. Đường dẫn Python hiển thị cần trỏ vào `.venv`. Nếu notebook nằm trong thư mục con, điều chỉnh đường dẫn CSV theo thư mục làm việc của kernel.

**Notebook 02–04 và các hàm huấn luyện hiện là khung để hoàn thiện.** Bắt đầu bằng cell audit trong notebook 01; chỉ triển khai và chạy mô hình khi có dữ liệu hồ sơ đủ đặc trưng. Các notebook đã có phần thiết lập đường dẫn để import `src/` khi mở từ thư mục gốc hoặc `notebooks/`.

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
