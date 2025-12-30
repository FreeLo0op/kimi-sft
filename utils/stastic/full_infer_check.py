import json
import re
import sys
import os
import random

def check_inference_results(file_path):
    # Regex to extract the JSON-like parts from the text
    # Pattern for phoneme results: matches content inside [] after "音素评测结果："
    phoneme_pattern = re.compile(r"音素评测结果：(\[.*?\])(?:。|$)")
    # Pattern for word results: matches content inside [] after "单词评测结果："
    word_pattern = re.compile(r"单词评测结果：(\[.*?\])(?:。|$)")

    total_samples = 0
    phoneme_consistent_count = 0
    length_consistent_count = 0
    
    phoneme_inconsistent_audios = []
    length_inconsistent_audios = []

    print(f"Processing file: {file_path}")
    max_word_len = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Error decoding JSON at line {line_idx + 1}")
                continue

            total_samples += 1
            
            label_str = data.get('label', '')
            predict_str = data.get('predict', '')
            audio_path = data.get('audio', '')

            # --- 1. Phoneme Consistency Check ---
            # Check if the phonemes used in prediction match the phonemes in the label
            
            label_phn_match = phoneme_pattern.search(label_str)
            predict_phn_match = phoneme_pattern.search(predict_str)
            
            is_phn_consistent = False
            label_phns = []
            predict_phns = []
            
            if label_phn_match and predict_phn_match:
                try:
                    label_phn_list = json.loads(label_phn_match.group(1))
                    predict_phn_list = json.loads(predict_phn_match.group(1))
                    
                    # Extract just the phoneme strings to compare
                    label_phns = [item.get('phn') for item in label_phn_list]
                    predict_phns = [item.get('phn') for item in predict_phn_list]
                    
                    if label_phns == predict_phns:
                        is_phn_consistent = True
                        phoneme_consistent_count += 1
                except json.JSONDecodeError:
                    print(f"Error parsing phoneme JSON at line {line_idx + 1}")
                    pass
            
            if not is_phn_consistent:
                phoneme_inconsistent_audios.append({
                    "line": line_idx + 1,
                    "audio": audio_path,
                    "label_phns": label_phns,
                    "predict_phns": predict_phns
                })

            # --- 2. Length Consistency Check ---
            # Check if the number of word scores in prediction matches the number of words in label
            
            # Determine expected word count from label
            # Priority 1: Length of phoneme list (most accurate if parsed correctly)
            label_word_count = -1
            if label_phn_match:
                try:
                    label_phn_list = json.loads(label_phn_match.group(1))
                    label_word_count = len(label_phn_list)
                    max_word_len = max(max_word_len, label_word_count)
                except:
                    pass
            
            # Priority 2: "共X个单词" text in label
            if label_word_count == -1:
                count_match = re.search(r"共(\d+)个单词", label_str)
                if count_match:
                    label_word_count = int(count_match.group(1))

            # Get predicted word result length
            predict_word_match = word_pattern.search(predict_str)
            predict_word_result_len = -1
            
            if predict_word_match:
                try:
                    predict_word_list = json.loads(predict_word_match.group(1))
                    predict_word_result_len = len(predict_word_list)
                except:
                    pass
            
            if label_word_count != -1 and predict_word_result_len != -1:
                if label_word_count == predict_word_result_len:
                    length_consistent_count += 1
                else:
                    length_inconsistent_audios.append({
                        "line": line_idx + 1,
                        "audio": audio_path,
                        "label_len": label_word_count,
                        "predict_len": predict_word_result_len
                    })
            else:
                length_inconsistent_audios.append({
                        "line": line_idx + 1,
                        "audio": audio_path,
                        "reason": "Could not extract lengths",
                        "label_len_extracted": label_word_count,
                        "predict_len_extracted": predict_word_result_len
                    })

    if total_samples == 0:
        print("No samples found.")
        return

    # Output results
    print("-" * 50)
    print(f"Total Samples: {total_samples}")
    
    print("-" * 50)
    print(f"Phoneme Consistency Rate: {phoneme_consistent_count / total_samples:.2%} ({phoneme_consistent_count}/{total_samples})")
    print(f"Number of Phoneme Inconsistent Audios: {len(phoneme_inconsistent_audios)}")
    
    print("-" * 50)
    print(f"Length Consistency Rate: {length_consistent_count / total_samples:.2%} ({length_consistent_count}/{total_samples})")
    print(f"Number of Length Inconsistent Audios: {len(length_inconsistent_audios)}")

    # Save inconsistent audios to file
    output_dir = os.path.dirname(file_path)
    # Or save in current directory
    output_dir = os.getcwd()
    
    phn_out_path = os.path.join(output_dir, "phoneme_inconsistent.json")
    len_out_path = os.path.join(output_dir, "length_inconsistent.json")

    with open(phn_out_path, "w", encoding="utf-8") as f:
        json.dump(phoneme_inconsistent_audios, f, indent=2, ensure_ascii=False)
        
    with open(len_out_path, "w", encoding="utf-8") as f:
        json.dump(length_inconsistent_audios, f, indent=2, ensure_ascii=False)
        
    print("-" * 50)
    print(f"Inconsistent details saved to:\n{phn_out_path}\n{len_out_path}")

    random.shuffle(lines)
    for line in lines[:20]:
        line = json.loads(line)
        label, predict = line.get('label', ''), line.get('predict', '')
        print(f'{label}\t{predict}')
    
    print(f"Max word length encountered in labels: {max_word_len}")

if __name__ == "__main__":
    # Default path
    default_path = "/mnt/pfs_l2/jieti_team/SFT/hupeng/mdd_lm/saves/post_training/base_model_v5/multi_pa_v15_1/infers/model3_tal-k12_full_pa/generated_predictions.jsonl"
    
    file_path = default_path
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
        
    check_inference_results(file_path)
    
