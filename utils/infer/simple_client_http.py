#!/usr/bin/env python3
"""
Simple HTTP client for Kimi Audio Triton service.
- Single-process, single-request example (optional repeat).
- No batching support (max_batch_size=0 in ensemble).
"""

import argparse
import wave
import uuid

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype, InferenceServerException

PROMPT_TEMPLATE = (
    "根据音频和评测文本，评测句子整体发音准确性，评分按顺序分为a,b,c,d到k共11档，"
    "a档表示0分，k档表示10分，每个档跨度是1分，最后输出档次。评测文本：{}"
)


def load_audio(audio_path: str):
    with wave.open(audio_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    return audio_bytes, sample_rate


def build_inputs(text: str, audio_bytes: bytes, sample_rate: int):
    """Build input tensors for Triton inference (no batch dimension)"""
    # Shape: [1] for text and sample_rate, [audio_len] for audio
    text_np = np.array([text.encode("utf-8")], dtype=np.object_)
    audio_np = np.frombuffer(audio_bytes, dtype=np.uint8)
    sample_rate_np = np.array([sample_rate], dtype=np.int32)

    inputs = [
        httpclient.InferInput("TEXT_CONTENT", text_np.shape, np_to_triton_dtype(text_np.dtype)),
        httpclient.InferInput("AUDIO_DATA", audio_np.shape, "UINT8"),
        httpclient.InferInput("SAMPLE_RATE", sample_rate_np.shape, "INT32"),
    ]
    inputs[0].set_data_from_numpy(text_np, binary_data=True)
    inputs[1].set_data_from_numpy(audio_np, binary_data=True)
    inputs[2].set_data_from_numpy(sample_rate_np, binary_data=True)
    return inputs


def infer_once(client: httpclient.InferenceServerClient, model_name: str, text: str, audio_bytes: bytes, sample_rate: int):
    """
    Perform one inference request.
    Returns: (token_id, pron_score, error_msg, request_id)
    If successful: (token_id, pron_score, None, request_id)
    If failed: (None, None, error_message, request_id)
    """
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    try:
        inputs = build_inputs(text, audio_bytes, sample_rate)
        outputs = [
            httpclient.InferRequestedOutput("OUTPUT_TOKEN_ID", binary_data=True),
            httpclient.InferRequestedOutput("PRON_SCORE", binary_data=True),
        ]
        result = client.infer(model_name=model_name, inputs=inputs, outputs=outputs, request_id=request_id)
        token_id = int(result.as_numpy("OUTPUT_TOKEN_ID").flatten()[0])
        pron_score = float(result.as_numpy("PRON_SCORE").flatten()[0])
        return token_id, pron_score, None, request_id
    except InferenceServerException as e:
        # Triton server returned an error
        # e.status() returns HTTP status code (400, 404, 500, 503, etc.)
        # e.message() returns error details
        error_msg = f"HTTP {e.status()}: {e.message()}"
        return None, None, error_msg, request_id
    except Exception as e:
        # Other errors (network, parsing, etc.)
        error_msg = f"Client error: {str(e)}"
        return None, None, error_msg, request_id


def main():
    parser = argparse.ArgumentParser(description="Simple HTTP client for Kimi Audio Triton")
    parser.add_argument("--url", type=str, default="localhost:8010", help="Triton HTTP server URL")
    parser.add_argument("--model", type=str, default="kimi_ensemble", help="Model name")
    parser.add_argument("--audio", type=str, required=True, help="Path to wav audio file")
    parser.add_argument("--text", type=str, required=True, help="Evaluation text")
    parser.add_argument("--use_prompt", action="store_true", help="Wrap text with prompt template")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat requests (reuses same client)")
    args = parser.parse_args()

    text = PROMPT_TEMPLATE.format(args.text) if args.use_prompt else args.text
    audio_bytes, sample_rate = load_audio(args.audio)

    client = httpclient.InferenceServerClient(url=args.url, verbose=False)

    if not client.is_server_live() or not client.is_model_ready(args.model):
        raise RuntimeError(f"Server not ready or model '{args.model}' not loaded")

    success_count = 0
    error_count = 0

    for i in range(max(args.repeat, 1)):
        token_id, pron_score, error, request_id = infer_once(client, args.model, text, audio_bytes, sample_rate)
        if error is None:
            print(f"[{i+1}] ✓ request_id={request_id}, token_id={token_id}, pron_score={pron_score}")
            success_count += 1
        else:
            print(f"[{i+1}] ✗ request_id={request_id}, Error: {error}")
            error_count += 1

    if args.repeat > 1:
        print(f"\nSummary: {success_count} success, {error_count} errors")


if __name__ == "__main__":
    main()
