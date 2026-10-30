# GAMA: A Multi-graph-based Anomaly Detection Framework for Business Processes via Graph Neural Networks

This is the source code of the paper '[**GAMA: A Multi-graph-based Anomaly Detection Framework for Business Processes via Graph Neural Networks**](https://www.techrxiv.org/doi/full/10.36227/techrxiv.23627850.v2)'.

![model](pic/architecture.png)

> **Ghi chú:** Repo này là bản fork của [guanwei49/GAMA](https://github.com/guanwei49/GAMA), dùng cho bài tập lớn cuối kỳ môn Cơ sở các hệ thống thông tin (Trường Đại học Công nghệ, ĐHQGHN) của Hoàng Việt Hùng (MSV 24022658). Các phần từ đầu đến hết mục *Experiment Results (bài gốc)* là nội dung gốc của tác giả. Phần đóng góp và thực nghiệm của nhóm nằm ở mục **Thực nghiệm của nhóm** phía dưới.

## Requirements

- [PyTorch==1.13.0](https://pytorch.org)
- [NumPy==1.23.5](https://numpy.org)
- [scikit-learn==1.2.1](https://scikit-learn.org)
- [pm4py==2.6.1](https://pm4py.fit.fraunhofer.de/)
- [PyG==2.1.0](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
- [pandas==1.5.3](https://pandas.pydata.org/)

## Using Our Code

```
    python main.py --mode eval --TF FAP
```

Two modes have been implemented:

- *eval*: Utilizing the anomalous event logs located in the ***eventlogs*** folder to obtain evaluation results (For reproducibility of the experiments).
- *test*: Detecting anomalies in the event log with the 'xes' format and obtaining anomaly detection results (For practical application).

Three teacher forcing (TF) styles have been implemented:

- *AN*: We consider that the current attribute value depends mainly on the current activity name. Therefore, at current event , the ground truth activity name is used to guide the prediction of the probability distribution.
- *PAV*: We consider that the current attribute value depends mainly on the previous attribute value. Therefore, the previous ground truth attribute value is used to guide the prediction of the probability distribution.
- *FAP*: We consider that the current attribute value depends both on the current activity name and the previous attribute value. Therefore, the fusion of current ground truth activity name and the previous ground truth attribute value is used to guide the prediction of the probability distribution.

## Datasets

Six commonly used real-life logs:

i) **[*Billing*](https://doi.org/10.4121/uuid:76c46b83-c930-4798-a1c9-4be94dfeb741)**: This log contains events that pertain to the billing of medical services provided by a hospital.

ii) **[*Receipt*](https://doi.org/10.4121/12709127.v2)**: This log contains records of the receiving phase of the building permit application process in an anonymous municipality.

iii) **[*Sepsis*](https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460)**: This log contains events of sepsis cases from a hospital.

iv) **[*RTFMP*](https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5)**: Real-life event log of an information system managing road traffic fines.

v) **[*Permit*](https://doi.org/10.4121/uuid:ea03d361-a7cd-4f5e-83d8-5fbdf0362550)**: it contains events related to travel permits (including all related events of relevant prepaid travel cost declarations and travel declarations).

vi) **[*Declaration*](https://doi.org/10.4121/uuid:2bbf8f6a-fc50-48eb-aa9e-c4ea5ef7e8c5)**: it contains events related to international travel declarations.

Eight synthetic logs: i.e., ***Paper***, ***P2P***, ***Small***, ***Medium***, ***Large***, ***Huge***, ***Gigantic***, and ***Wide***.

The summary of statistics for each event log is presented below:

| Log         | #Activities | #Traces | #Events     | Max trace length | Min trace length | #Attributes | #Attribute values |
|:-----------:|:-----------:|:-------:|:-----------:|:----------------:|:----------------:|:-----------:|:-----------------:|
| Gigantic    | 76-78       | 5000    | 28243-31989 | 11               | 3                | 1-4         | 70-363            |
| Huge        | 54          | 5000    | 36377-42999 | 11               | 5                | 1-4         | 69-340            |
| Large       | 42          | 5000    | 51099-56850 | 12               | 10               | 1-4         | 68-292            |
| Medium      | 32          | 5000    | 28416-31372 | 8                | 3                | 1-4         | 66-276            |
| P2p         | 13          | 5000    | 37941-42634 | 11               | 7                | 1-4         | 39-146            |
| Paper       | 14          | 5000    | 49839-54390 | 12               | 9                | 1-4         | 36-128            |
| Small       | 20          | 5000    | 42845-46060 | 10               | 7                | 1-4         | 39-144            |
| Wide        | 23-34       | 5000    | 29128-31228 | 7                | 5-6              | 1-4         | 53-264            |
| Billing     | 18          | 100000  | 451359      | 217              | 1                | 0           | 0                 |
| Receipt     | 27          | 1434    | 8577        | 25               | 1                | 2           | 58                |
| Sepsis      | 16          | 1050    | 15214       | 185              | 3                | 1           | 26                |
| RTFMP       | 11          | 150370  | 561470      | 20               | 2                | 0           | 0                 |
| Permit      | 51          | 7065    | 86581       | 90               | 3                | 2           | 10                |
| Declaration | 34          | 6449    | 72151       | 27               | 3                | 2           | 10                |

Logs containing artificial anomalies ranging from 5% to 45% are stored in the folder '***eventlogs***'. The file names are formatted as *log_name*-*anomaly_ratio*-*ID*.

## Experiment Results (bài gốc)

Critical difference diagram over trace-level anomaly detection:

![cd_trace](pic/cd_trace.png)

Critical difference diagram over attribute-level anomaly detection:

![cd_attr](pic/cd_attr.png)

*F−scores* over synthetic logs where 'T' and 'A' represent trace- and attribute-level anomaly detection respectively.

|          | Paper     | Paper     | P2P       | P2P       | Small     | Small     | Medium    | Medium    | Large     | Large     | Huge      | Huge      | Gigantic  | Gigantic  | Wide      | Wide      |
|:--------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|          | T         | A         | T         | A         | T         | A         | T         | A         | T         | A         | T         | A         | T         | A         | T         | A         |
| OC-SVM   | 0.498     | -         | 0.480     | -         | 0.522     | -         | 0.446     | -         | 0.480     | -         | 0.446     | -         | 0.462     | -         | 0.460     | -         |
| Naive    | 0.866     | -         | 0.850     | -         | 0.898     | -         | 0.691     | -         | 0.715     | -         | 0.690     | -         | 0.574     | -         | 0.779     | -         |
| Sampling | 0.901     | -         | 0.886     | -         | 0.896     | -         | 0.860     | -         | 0.910     | -         | 0.890     | -         | 0.800     | -         | 0.888     | -         |
| GAE      | 0.472     | -         | 0.559     | -         | 0.468     | -         | 0.449     | -         | 0.530     | -         | 0.429     | -         | 0.434     | -         | 0.561     | -         |
| DAE      | 0.799     | 0.468     | 0.767     | 0.475     | 0.829     | 0.463     | 0.713     | 0.436     | 0.747     | 0.433     | 0.691     | 0.415     | 0.580     | 0.288     | 0.753     | 0.455     |
| VAE      | 0.828     | 0.190     | 0.655     | 0.212     | 0.788     | 0.219     | 0.637     | 0.230     | 0.772     | 0.201     | 0.589     | 0.213     | 0.495     | 0.181     | 0.640     | 0.230     |
| LAE      | 0.678     | 0.243     | 0.666     | 0.266     | 0.748     | 0.239     | 0.584     | 0.270     | 0.571     | 0.250     | 0.531     | 0.268     | 0.504     | 0.234     | 0.699     | 0.271     |
| BINet    | 0.543     | 0.330     | 0.557     | 0.342     | 0.566     | 0.358     | 0.521     | 0.319     | 0.549     | 0.333     | 0.526     | 0.331     | 0.525     | 0.320     | 0.551     | 0.345     |
| GAMA-AN  | **0.949** | **0.701** | **0.950** | **0.686** | **0.955** | **0.717** | **0.873** | **0.716** | **0.945** | **0.768** | **0.916** | **0.763** | **0.821** | **0.701** | **0.921** | **0.724** |
| GAMA-PAV | **0.976** | **0.675** | **0.974** | **0.664** | **0.981** | **0.663** | **0.903** | **0.654** | **0.944** | **0.678** | **0.909** | **0.663** | **0.809** | **0.614** | **0.950** | **0.670** |
| GAMA-FAP | **0.955** | **0.699** | **0.949** | **0.683** | **0.955** | **0.708** | **0.872** | **0.700** | **0.947** | **0.752** | **0.922** | **0.750** | **0.833** | **0.691** | **0.923** | **0.712** |

*F−scores* over real-life logs where 'T' and 'A' represent trace- and attribute-level anomaly detection respectively.

|          | Billing   | Billing   | Receipt   | Receipt   | Sepsis    | Sepsis    | RTFMP     | RTFMP     | Permit    | Permit    | Declaration | Declaration |
|:--------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:-----------:|:-----------:|
|          | T         | A         | T         | A         | T         | A         | T         | A         | T         | A         | T           | A           |
| OC-SVM   | 0.340     | -         | 0.464     | -         | 0.415     | -         | 0.507     | -         | 0.405     | -         | 0.449       | -           |
| Naive    | 0.668     | -         | 0.638     | -         | 0.392     | -         | 0.776     | -         | 0.462     | -         | 0.495       | -           |
| Sampling | 0.701     | -         | 0.647     | -         | 0.391     | -         | 0.721     | -         | 0.458     | -         | 0.507       | -           |
| GAE      | 0.385     | -         | 0.420     | -         | 0.391     | -         | 0.341     | -         | 0.386     | -         | 0.406       | -           |
| DAE      | 0.754     | 0.444     | 0.650     | 0.158     | 0.461     | 0.136     | 0.865     | 0.498     | 0.522     | 0.182     | 0.576       | 0.201       |
| VAE      | 0.731     | 0.435     | 0.524     | 0.134     | 0.448     | 0.172     | 0.813     | 0.517     | 0.484     | 0.188     | 0.476       | 0.180       |
| LAE      | 0.784     | 0.509     | 0.526     | 0.218     | 0.408     | 0.126     | 0.874     | 0.505     | 0.486     | 0.287     | 0.514       | 0.345       |
| BINet    | 0.621     | 0.442     | 0.575     | 0.416     | 0.435     | 0.192     | 0.744     | 0.493     | 0.641     | 0.423     | 0.678       | **0.499**   |
| GAMA-AN  | **0.792** | **0.545** | **0.763** | **0.548** | **0.570** | **0.457** | **0.899** | **0.534** | **0.682** | **0.428** | **0.727**   | 0.461       |
| GAMA-PAV | **0.791** | **0.544** | **0.778** | **0.538** | **0.510** | **0.376** | **0.914** | **0.574** | 0.634     | 0.369     | 0.669       | 0.406       |
| GAMA-FAP | **0.811** | **0.548** | **0.752** | **0.535** | **0.573** | **0.450** | **0.914** | **0.576** | **0.679** | 0.402     | **0.718**   | 0.444       |

---

# Thực nghiệm của nhóm

Phần này mô tả những gì đã thêm vào repo gốc và kết quả tái lập.

## Thay đổi so với repo gốc

| Thay đổi | Mô tả |
|---|---|
| `main.py` | Thêm tham số dòng lệnh `--only`, `--hidden_dim`, `--n_epochs`, `--lr`, `--seed`, `--res` (trước đây đặt cứng trong code) |
| `scripts/run_all.py` | Chạy 3 log (Paper, Small, Sepsis) x 3 kiểu TF (AN, PAV, FAP) với `seed = 42`, `20` epoch; kết quả vào `results/`, log chạy vào `logs/` |
| `scripts/summary.py` | Tổng hợp kết quả chính thành `results/summary.csv` |
| `scripts/summary_khac.py` | Tổng hợp các kịch bản khác tác giả thành `results/summary_khac.csv` |
| `results/`, `logs/` | Kết quả và log huấn luyện của nhóm |

Các tham số dòng lệnh của `main.py`:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--mode` | `eval` | `eval` hoặc `test` (xem phần *Using Our Code*) |
| `--TF` | `FAP` | Kiểu teacher forcing: `AN`, `PAV`, `FAP` |
| `--beta` | `0.01` | Ngưỡng lọc cạnh của đồ thị toàn cục |
| `--only` | (rỗng) | Chỉ chạy các log có tên chứa chuỗi này |
| `--hidden_dim` | `64` | Kích thước lớp ẩn của GAT và GRU |
| `--n_epochs` | `20` | Số epoch |
| `--lr` | `0.0005` | Learning rate khởi đầu |
| `--seed` | không đặt | Seed ngẫu nhiên |
| `--res` | `result.csv` | File lưu kết quả |

## Cách chạy lại

Một cấu hình:

```
python main.py --mode eval --TF FAP --only paper-0.30-1 --seed 42 --n_epochs 20 --res results\paper-0.30-1_FAP.csv
```

Toàn bộ thực nghiệm chính (script bỏ qua cấu hình đã có file kết quả):

```
python scripts\run_all.py
python scripts\summary.py
```

Các kịch bản khác tác giả được chạy thủ công bằng `main.py` trên log Paper với TF = FAP, ví dụ:

```
python main.py --mode eval --TF FAP --only paper-0.30-1 --hidden_dim 16 --res results\khac\paper_hid16.csv
```

Seed của các lần chạy thủ công này không được ghi lại, nên khi chạy lại kết quả có thể lệch nhẹ (xem mục độ ổn định).

## Cách đánh giá

Với mỗi mức (trace, event, attribute), điểm bất thường được so với nhãn thật bằng `precision_recall_curve`. Precision, Recall, F1 lấy tại ngưỡng cho F1 cao nhất (giống grid search ngưỡng trong bài báo), kèm Average Precision (AP, không phụ thuộc ngưỡng). Vì ngưỡng chọn bằng nhãn thật nên F1 là mức tốt nhất có thể đạt, chưa phản ánh việc triển khai khi không có nhãn. Xem `utils/eval.py`.

## Kết quả

Cấu hình chung: `lr = 0.0005`, `hidden_dim = 64`, `beta = 0.01`, `20` epoch, AP = 0.30, `seed = 42` (trừ khi ghi khác).

### Kịch bản 1: Chạy lại như tác giả (3 kiểu teacher forcing)

| Log | TF | F1 (T) | F1 (A) | Thời gian (s) |
|---|---|---|---|---|
| Paper | AN | 0.9797 | **0.7602** | 289 |
| Paper | PAV | **0.9971** | 0.7074 | 282 |
| Paper | FAP | 0.9908 | 0.7511 | 347 |
| Small | AN | 0.9654 | **0.7762** | 275 |
| Small | PAV | **0.9976** | 0.6730 | 308 |
| Small | FAP | 0.9753 | 0.7746 | 299 |
| Sepsis | AN | **0.575** | 0.447 | 2189 |
| Sepsis | PAV | 0.537 | 0.352 | 2603 |
| Sepsis | FAP | 0.560 | **0.456** | 2535 |

- PAV cho F1 mức trace cao nhất trên log tổng hợp, AN cho F1 mức attribute cao nhất, FAP nằm giữa. Xu hướng này khớp với bài báo.
- Trên Sepsis, PAV thấp nhất ở cả hai mức, cũng giống bài báo. Sepsis khó hơn Paper và Small.
- Sepsis chạy lâu hơn dù ít trace hơn, vì trace dài (tối đa 185 event).
- Không so trực tiếp được với bảng của bài báo, vì bài báo lấy trung bình trên nhiều mức AP và nhiều biến thể log, còn ở đây chỉ chạy một log ở AP = 0.30.

Ghi chú: `results/summary.csv` có thể còn giữ kết quả Paper-FAP của lần chạy đầu tiên khi chưa cố định seed (trace 0.9862, attribute 0.7522). Bảng trên dùng lần chạy seed 42.

### Kịch bản 2: Khác tác giả (log Paper, TF = FAP)

| Thay đổi | F1 (T) | F1 (A) | Thời gian (s) |
|---|---|---|---|
| Cấu hình cơ sở (seed 42) | 0.9908 | 0.7511 | 347 |
| AP = 0.10 | 1.0000 | 0.8175 | 405 |
| AP = 0.45 | 0.9895 | 0.7422 | 324 |
| beta = 0.001 | 0.9795 | 0.7412 | 310 |
| beta = 0.1 | 0.9944 | 0.7558 | 306 |
| hidden_dim = 16 | 0.9574 | **0.8135** | 150 |
| hidden_dim = 256 | 0.9873 | 0.7068 | 3031 |

- F1 mức attribute giảm khi tỷ lệ bất thường tăng (0.8175, 0.7511, 0.7422 ứng với AP = 0.10, 0.30, 0.45).
- beta ít ảnh hưởng: F1 dao động nhỏ hơn 0.02, phù hợp nhận xét của tác giả rằng mô hình không nhạy với beta.
- F1 mức attribute cao nhất ở hidden_dim = 16 và giảm khi tăng lên 256, trong khi thời gian chạy tăng khoảng 20 lần. Giả thuyết: mô hình lớn tái tạo luôn cả mẫu bất thường (khớp lập luận ở mục 5.10.2 của bài). Mới có 3 điểm và 1 seed nên chưa đủ để kết luận.

### Độ ổn định theo seed (Paper, FAP)

| Lần chạy | F1 (T) | F1 (A) |
|---|---|---|
| Seed 42 | 0.9908 | 0.7511 |
| Seed 1 | 0.9948 | 0.7498 |
| Seed 2 | 0.9915 | 0.7507 |
| Không đặt seed (lần chạy đầu) | 0.9862 | 0.7522 |

F1 mức attribute chỉ dao động trong khoảng 0.0024.

### Điểm lệch so với bài báo

Bài báo (mục 5.2) ghi learning rate là 0.0002, nhưng code công bố dùng 0.0005 (xác nhận qua log huấn luyện). Nhóm giữ nguyên cấu hình trong code và chưa chạy lại với 0.0002.

## To Cite Our Paper

```
@article{guan2024gama,
  title={GAMA: A multi-graph-based anomaly detection framework for business processes via graph neural networks},
  author={Guan, Wei and Cao, Jian and Gu, Yang and Qian, Shiyou},
  journal={Information Systems},
  volume={124},
  pages={102405},
  year={2024},
  publisher={Elsevier}
}
```
