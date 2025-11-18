import os
import re
import sys
import json
from sklearn.metrics import classification_report, confusion_matrix

def extract_predicted_labels(item:str):
    data = json.loads(item)
    label = data.get("label", None)
    predict = data.get("predict", None)
    return int(label), int(predict)
    pass

def analyze_audio_detection_results(y_true, y_pred, labels):
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

def main(
        infer_result_file: str
    ):
    y_true, y_pred = [], []
    with open(infer_result_file, "r") as f:
        for line in f:
            label, predict = extract_predicted_labels(line)
            if label is not None and predict is not None:
                y_true.append(label)
                y_pred.append(predict)
            else:
                print(f"Skipping line due to missing label or predict: {line.strip()}")

    labels = ["Class 0", "Class 1"]  # Replace with your actual class names
    analyze_audio_detection_results(y_true, y_pred, labels)

if __name__ == "__main__":
    infer_result_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v2.5/model_infer_2/infer_res/infer_prompt_data_test.jsonl'
    main(infer_result_file)