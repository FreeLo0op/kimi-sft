from kimia_infer.api.kimia import KimiAudio
import os
import json
import soundfile as sf
import argparse
from tqdm import tqdm

sampling_params = {
    "audio_temperature": 0.8,
    "audio_top_k": 1,
    "text_temperature": 0.0,
    "text_top_k": 1,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

def prompt_loader(prompt_path:str, if_convert:bool=True) -> list:
    infer_messages = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        if if_convert:
            f = json.load(f)
            for item in f:
                single_messages = []
                messages = item["messages"]
                system_content = messages[0]["content"]
                user_content = messages[1]["content"]
                user_content = user_content.replace("<audio>", "").strip()
                infer_content = f"{system_content}{user_content}"

                label = item["messages"][2]["content"]
                audio = item["audios"][0]

                single_messages.append({"role": "user", "message_type": "text", "content": infer_content})
                single_messages.append({"role": "user", "message_type": "audio", "content": audio})
                # single_messages.append({"role": "user", "message_type": "audio-text", "content": [audio, infer_content]})
                single_messages.append({"role": "assistant_gt", "message_type": "text", "content": label})
                infer_messages.append(single_messages)
        else:
            for line in f:
                item = json.loads(line)
                single_messages = item['conversation']
                infer_messages.append(single_messages)
    return infer_messages

def main(
        infer_prompt:str,
        output_path:str,
        gpu_id:str,
        model_path:str="/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/Kimi_Pa_V1.1_hf_for_inference",
    ):
    infer_messages = prompt_loader(infer_prompt, if_convert=True)
    
    model = KimiAudio(model_path=model_path, load_detokenizer=False, device=f'cuda:{gpu_id}')
    # infer_res = []
    fo = open(output_path, "w", encoding="utf-8")
    for i in tqdm(range(len(infer_messages)), desc="Inference", disable=False):
        messages = infer_messages[i][:-1]
        label = infer_messages[i][-1]["content"]
        audio = infer_messages[i][1]["content"]
        # audio, label = infer_messages[i][0]["content"]

        text = model.generate(messages, **sampling_params, output_type="text")
        # wav, text = model.generate(messages, **sampling_params, output_type="both")

        infer_res = {
            "prompt": messages[0]["content"],
            "audio": audio,
            "label": label,
            "predict": text,
        }
        fo.write(json.dumps(infer_res, ensure_ascii=False) + "\n")
        fo.flush()
    fo.close()
    # with open(output_path, "w", encoding="utf-8") as fo:
    #     for line in infer_res:
    #         fo.write(json.dumps(line, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--model_path", type=str, required=True, help="Path to the finetuned model")
    argparse.add_argument("--infer_prompt", type=str, required=True, help="Path to the inference prompt file")
    argparse.add_argument("--output_path", type=str, required=True, help="Path to save the inference results")
    argparse.add_argument("--gpu_id", type=str, required=True, help="GPU id to use")
    args = argparse.parse_args()

    model_path = args.model_path
    infer_prompt_file = args.infer_prompt
    output_path = args.output_path
    gpu_id = args.gpu_id
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # infer_prompt_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/multi_task/sft/test/tal-k12_sent_pa_accuracy_nocot-v2_test.json'
    # output_path = '/mnt/pfs_l2/jieti_team/SFT/hupeng/github/Kimi-Audio/output/infer_res/infer_tal-k12_sent_pa_accuracy_nocot-v2_test.json'

    main(
        infer_prompt=infer_prompt_file,
        output_path=output_path,
        gpu_id=gpu_id,
        model_path=model_path
    )