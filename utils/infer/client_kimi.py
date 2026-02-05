#!/usr/bin/env python3
"""
Kimi Audio 发音评分客户端
用于向 Triton Server 发送音频数据进行发音评分推理
"""
import os
import numpy as np
import tritonclient.grpc as grpcclient
from tqdm import tqdm
import concurrent.futures

# ============ 配置 ============
TRITON_URL = "10.157.14.49:8011"
MODEL_NAME = "kimi_ensemble"

# 评分 prompt 模板
PROMPT_TEMPLATE = "根据音频和评测文本，评测句子整体发音准确性，评分按顺序分为a,b,c,d到k共11档，a档表示0分，k档表示10分，每个档跨度是1分，最后输出档次。评测文本：{text}"


# ============ 工具函数 ============
def load_audio(audio_path: str) -> tuple[bytes, int]:
    """加载音频文件，返回字节数据和采样率"""
    import wave
    with wave.open(audio_path, 'rb') as f:
        sample_rate = f.getframerate()
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()
    return audio_bytes, sample_rate


def create_client(url: str = TRITON_URL):
    """创建 Triton gRPC 客户端"""
    return grpcclient.InferenceServerClient(url=url)

def load_text(file:str):
    text_dict = {}
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split("\t")
        key, text = line[0], line[1]
        if key in text_dict:
            print(f"Warning: Duplicate key found: {key}")
        text_dict[key] = text
    print(f'Total samples for inference: {len(text_dict)}')
    return text_dict

def load_audios(file:str):
    audio_dict = {}
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        key, audio_path = line.strip().split("\t", maxsplit=1)
        audio_dict[key] = audio_path
    return audio_dict

def data_unique(audio_dict: dict, text_dict: dict):
    shared_key = set(audio_dict.keys()) & set(text_dict.keys())
    audio_dict = {k: audio_dict[k] for k in shared_key}
    text_dict = {k: text_dict[k] for k in shared_key}
    return audio_dict, text_dict

# ============ 推理函数 ============
def infer(client, audio_bytes: bytes, sample_rate: int, eval_text: str) -> dict:
    """
    发送推理请求
    
    Args:
        client: Triton 客户端
        audio_bytes: 音频字节数据
        sample_rate: 采样率
        eval_text: 评测文本
    
    Returns:
        {"token_id": int, "pron_score": int}
    """
    # 构建 prompt
    prompt = PROMPT_TEMPLATE.format(text=eval_text)
    
    # 准备输入
    text_np = np.array([prompt.encode("utf-8")], dtype=np.object_)
    audio_np = np.frombuffer(audio_bytes, dtype=np.uint8)
    # audio_np = np.expand_dims(audio_np, axis=0)  # 添加 batch 维度
    sr_np = np.array([sample_rate], dtype=np.int32)
    
    inputs = [
        grpcclient.InferInput("TEXT_CONTENT", text_np.shape, "BYTES"),
        grpcclient.InferInput("AUDIO_DATA", audio_np.shape, "UINT8"),
        grpcclient.InferInput("SAMPLE_RATE", sr_np.shape, "INT32"),
    ]
    inputs[0].set_data_from_numpy(text_np)
    inputs[1].set_data_from_numpy(audio_np)
    inputs[2].set_data_from_numpy(sr_np)
    
    outputs = [
        grpcclient.InferRequestedOutput("OUTPUT_TOKEN_ID"),
        grpcclient.InferRequestedOutput("PRON_SCORE"),
    ]
    
    # 发送请求
    result = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
    
    return {
        "token_id": int(result.as_numpy("OUTPUT_TOKEN_ID")[0]),
        "pron_score": int(result.as_numpy("PRON_SCORE")[0]),
    }

def process_item(client, key, audio_path, eval_text):
    try:
        audio_bytes, sample_rate = load_audio(audio_path)
        result = infer(client, audio_bytes, sample_rate, eval_text)
        return key, result['pron_score']
    except Exception as e:
        print(f"Error processing {key}: {e}")
        return key, None

# ============ 主函数 ============
def main():
    # 示例数据
    audio_path = "/home/haozhiqiang/repos/tal_kimi/data/wav_snt/16991423581951170633426500337664.wav"
    audio_path = "/mnt/vepfs/jieti_team/SFT/hupeng/data/tal-k12/wavs/wav_snt/16991423581951170633426500337664.wav"
    audio_path = "./16991423581951170633426500337664.wav"
    audio_path = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/wavs/wav_snt/16991423581951170633426500337664.wav'
    eval_text = "my grandmother is young."
    
    # 加载音频
    audio_bytes, sample_rate = load_audio(audio_path)
    print(f"音频: {audio_path}")
    print(f"大小: {len(audio_bytes)} bytes, 采样率: {sample_rate} Hz")
    
    # 创建客户端并推理
    client = create_client()
    result = infer(client, audio_bytes, sample_rate, eval_text)
    
    # 输出结果
    print(f"\n评测文本: {eval_text}")
    print(f"Token ID: {result['token_id']}")
    print(f"发音得分: {result['pron_score']}")

def main2(
    infer_file:str,
    outpit_dir: str,
    max_workers: int = 4
    ):
    audio_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/wavpath_merged'
    audio_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/audio_detect/test/wavpath'
    text_dict = load_text(infer_file)
    audio_dict = load_audios(audio_file)
    audio_dict, text_dict = data_unique(audio_dict, text_dict)
    client = create_client()
    fo_path = os.path.join(outpit_dir, os.path.basename(infer_file))
    os.makedirs(outpit_dir, exist_ok=True)
    
    with open(fo_path, "w", encoding="utf-8") as fo:
        infer_res = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for key in text_dict.keys():
                futures.append(executor.submit(process_item, client, key, audio_dict[key], text_dict[key]))

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing samples"):
                key, score = future.result()
                if score is not None:
                    infer_res[key] = score
        for key, score in infer_res.items():
            fo.write(f"{key}\t{score}\n")
    print(f"Results saved to {fo_path}")

if __name__ == "__main__":
    files = [
        # '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_snt_score_batch1',
        # '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_snt_score_batch2',
        # '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_snt_score_batch4',
        '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_sent_score'
    ]
    # for file in files:
    #     main2(
    #         file,
    #         '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v3.3/infer_model/infer_res_client'
    #     )
    # main2(
    #     '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/audio_detect/test/label_snt_score',
    #     '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v3.3/infer_model/infer_res_client_batch',
    #     max_workers=2
    # )
    main()
