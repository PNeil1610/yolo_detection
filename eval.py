import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix
import seaborn as sns


MODEL_PATH = "F:/learning/Project_detection/runs/detect/train/weights/best.pt"
TEST_IMAGES = "F:/learning/Project_detection/dataset_split/test/images"
TEST_LABELS = "F:/learning/Project_detection/dataset_split/test/labels"

CLASSES = [
    "bmw", "mercedes", "vinfast", "toyota", "mitsubishi",
    "ford", "honda", "hyundai", "kia", "bien_so"
]

IOU_THRESHOLD = 0.5  


model = YOLO(MODEL_PATH)


y_true = []
y_pred = []

def compute_iou(box1, box2):  
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(box1Area + box2Area - interArea + 1e-6)

label_files = sorted(os.listdir(TEST_LABELS)) 

for label_file in label_files:
    label_path = os.path.join(TEST_LABELS, label_file)
    image_path = os.path.join(TEST_IMAGES, label_file.replace(".txt", ".jpg"))

    if not os.path.exists(image_path):
        image_path = image_path.replace(".jpg", ".png")
        if not os.path.exists(image_path):
            continue

    
    gt_boxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            c, x, y, w, h = map(float, line.split())
            gt_boxes.append((int(c), x, y, w, h))

    
    result = model.predict(source=image_path, conf=0.25, verbose=False)[0]
    pred_boxes = [(int(b.cls.item()), *b.xywhn.tolist()[0]) for b in result.boxes]

    matched_pred = set()


    for gt_class, gx, gy, gw, gh in gt_boxes:
        best_iou = 0
        best_pred = None


        for idx, (pc, px, py, pw, ph) in enumerate(pred_boxes):
            if idx in matched_pred:  # Đã dùng
                continue


            iou = compute_iou((gx-gw/2, gy-gh/2, gx+gw/2, gy+gh/2),
                              (px-pw/2, py-ph/2, px+pw/2, py+ph/2))
            
            if iou > best_iou:
                best_iou = iou
                best_pred = (idx, pc)

        if best_iou >= IOU_THRESHOLD:
            matched_pred.add(best_pred[0])
            y_true.append(gt_class)
            y_pred.append(best_pred[1])
        else:
         
            y_true.append(gt_class)
            y_pred.append(-1)


    for idx, (pc, _, _, _, _) in enumerate(pred_boxes):
        if idx not in matched_pred:
            y_true.append(-1)
            y_pred.append(pc)


cm_true = []
cm_pred = []

for t, p in zip(y_true, y_pred):
    if t != -1 and p != -1:
        cm_true.append(t)
        cm_pred.append(p)


cm = confusion_matrix(
    cm_true,
    cm_pred,
    labels=list(range(len(CLASSES)))
)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Only Valid Matches)")
plt.show()


from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# =========================
# 6. Tính Precision / Recall / F1
# =========================

# Dùng cm_true và cm_pred thay cho y_true_filtered / y_pred_filtered
y_true_filtered = cm_true
y_pred_filtered = cm_pred

precision_macro = precision_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
recall_macro = recall_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
f1_macro = f1_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)

precision_micro = precision_score(y_true_filtered, y_pred_filtered, average='micro', zero_division=0)
recall_micro = recall_score(y_true_filtered, y_pred_filtered, average='micro', zero_division=0)
f1_micro = f1_score(y_true_filtered, y_pred_filtered, average='micro', zero_division=0)

precision_weighted = precision_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
recall_weighted = recall_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
f1_weighted = f1_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)

print("\n===== METRICS =====")
print(f"Precision (Macro):  {precision_macro:.4f}")
print(f"Recall (Macro):     {recall_macro:.4f}")
print(f"F1-score (Macro):   {f1_macro:.4f}")

print(f"Precision (Micro):  {precision_micro:.4f}")
print(f"Recall (Micro):     {recall_micro:.4f}")
print(f"F1-score (Micro):   {f1_micro:.4f}")

print(f"Precision (Weighted):  {precision_weighted:.4f}")
print(f"Recall (Weighted):     {recall_weighted:.4f}")
print(f"F1-score (Weighted):   {f1_weighted:.4f}")

# =========================
# 7. Xem chi tiết từng class (nếu muốn)
# =========================
print("\n===== Classification Report =====")
print(classification_report(y_true_filtered, y_pred_filtered, target_names=CLASSES, zero_division=0))

