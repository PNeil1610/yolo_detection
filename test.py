import pandas as pd


df = pd.read_csv("runs/detect/train/results.csv")  


epoch_89 = df.loc[df["epoch"] == 89]

if not epoch_89.empty:
    precision = epoch_89["metrics/precision(B)"].values[0]
    recall = epoch_89["metrics/recall(B)"].values[0]
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("Chỉ số tại epoch 89:")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1_score)
else:
    print("Không tìm thấy epoch 89 trong file results.csv")
