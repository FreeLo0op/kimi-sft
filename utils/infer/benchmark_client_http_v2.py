#!/usr/bin/env python3
"""
High-Performance HTTP Benchmark Client for Kimi Audio Triton Service (Batch Version)
Features:
- Multi-process + async architecture to trigger server-side dynamic batching
- Uses concurrent async requests within each worker to fill server batches
- Modular design with separate data loading, inference, and statistics
- Detailed latency statistics (QPS, P50, P90, P99, etc.)

Note: Server-side batching is triggered by sending multiple concurrent requests.
The server's dynamic_batching config will combine them into batches.
"""

import argparse
import os
import sys
import time
import wave
import json
import uuid
import asyncio
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Tuple
import multiprocessing as mp
import threading
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# Use 'spawn' to avoid HTTP client issues with fork
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class TestCase:
    """Single test case with text and audio data"""
    key: str
    text: str
    audio_bytes: bytes
    sample_rate: int


@dataclass  
class InferenceResult:
    """Result of a single inference request"""
    request_id: int
    success: bool
    latency_ms: float
    token_id: Optional[int] = None
    pron_score: Optional[float] = None
    error: Optional[str] = None


@dataclass
class BenchmarkStats:
    """Aggregated benchmark statistics"""
    total_requests: int
    total_samples: int  # Total individual samples (requests * batch_size)
    successful_requests: int
    failed_requests: int
    total_time_sec: float
    qps: float  # Requests per second
    samples_per_sec: float  # Samples per second
    latencies_ms: List[float] = field(default_factory=list)
    
    @property
    def avg_latency(self) -> float:
        return np.mean(self.latencies_ms) if self.latencies_ms else 0.0
    
    @property
    def p50_latency(self) -> float:
        return np.percentile(self.latencies_ms, 50) if self.latencies_ms else 0.0
    
    @property
    def p90_latency(self) -> float:
        return np.percentile(self.latencies_ms, 90) if self.latencies_ms else 0.0
    
    @property
    def p95_latency(self) -> float:
        return np.percentile(self.latencies_ms, 95) if self.latencies_ms else 0.0
    
    @property
    def p99_latency(self) -> float:
        return np.percentile(self.latencies_ms, 99) if self.latencies_ms else 0.0
    
    @property
    def max_latency(self) -> float:
        return np.max(self.latencies_ms) if self.latencies_ms else 0.0
    
    @property
    def min_latency(self) -> float:
        return np.min(self.latencies_ms) if self.latencies_ms else 0.0


# =============================================================================
# Data Loading Module
# =============================================================================

class DataLoader:
    """Handles loading and preprocessing of test data"""
    
    PROMPT_TEMPLATE = (
        "根据音频和评测文本，评测句子整体发音准确性，评分按顺序分为a,b,c,d到k共11档，"
        "a档表示0分，k档表示10分，每个档跨度是1分，最后输出档次。评测文本：{}"
    )
    
    @staticmethod
    def load_audio_bytes(audio_path: str) -> Tuple[Optional[bytes], Optional[int]]:
        """Load audio file and return raw bytes and sample rate"""
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            return audio_bytes, sample_rate
        except Exception as e:
            print(f"[DataLoader] Error loading {audio_path}: {e}")
            return None, None
    
    @staticmethod
    def load_dataset(label_file: str, wavpath_file: str, max_items: Optional[int] = None) -> List[TestCase]:
        """Load dataset from label and wavpath files"""
        print(f"[DataLoader] Loading data from:")
        print(f"  Labels: {label_file}")
        print(f"  Audio paths: {wavpath_file}")
        
        # Load texts: key -> text
        texts = {}
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    texts[parts[0]] = parts[1]
        
        # Load wav paths: key -> path
        wav_paths = {}
        with open(wavpath_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and os.path.exists(parts[1]):
                    wav_paths[parts[0]] = parts[1]
        
        # Get intersection
        common_keys = list(set(texts.keys()) & set(wav_paths.keys()))
        print(f"[DataLoader] Found {len(texts)} texts, {len(wav_paths)} audio paths")
        print(f"[DataLoader] Valid intersection: {len(common_keys)}")
        
        if max_items and max_items > 0:
            common_keys = common_keys[:max_items]
        
        # Build test cases with preloaded audio
        print(f"[DataLoader] Preloading {len(common_keys)} audio files...")
        test_cases = []
        for i, key in enumerate(common_keys):
            audio_bytes, sample_rate = DataLoader.load_audio_bytes(wav_paths[key])
            if audio_bytes:
                test_cases.append(TestCase(
                    key=key,
                    text=DataLoader.PROMPT_TEMPLATE.format(texts[key]),
                    audio_bytes=audio_bytes,
                    sample_rate=sample_rate
                ))
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{len(common_keys)}...", end='\r')
        
        print(f"\n[DataLoader] Successfully loaded {len(test_cases)} test cases")
        return test_cases


# =============================================================================
# Triton HTTP Client Module (Single Request - Server Does Batching)
# =============================================================================

class TritonHTTPInferenceClient:
    """Handles Triton HTTP inference requests - single sample per request"""
    
    def __init__(self, url: str, model_name: str = "kimi_whole"):
        self.url = url
        self.model_name = model_name
        self._client = None
        self._httpclient = None
        self._np_to_triton_dtype = None
    
    def _ensure_connected(self):
        """Lazy initialization of HTTP client - important for multiprocessing"""
        if self._client is None:
            import tritonclient.http as httpclient
            from tritonclient.utils import np_to_triton_dtype
            self._httpclient = httpclient
            self._np_to_triton_dtype = np_to_triton_dtype
            self._client = httpclient.InferenceServerClient(url=self.url, verbose=False)
    
    def is_server_ready(self) -> bool:
        """Check if server is live and model is ready"""
        try:
            self._ensure_connected()
            live_ok = self._client.is_server_live()
            model_ready = self._client.is_model_ready(self.model_name)

            if not live_ok:
                print("[HTTPClient] /v2/health/live returned False")
            if not model_ready:
                print(f"[HTTPClient] /v2/models/{self.model_name}/ready returned False")

            return live_ok and model_ready
        except Exception as e:
            print(f"[HTTPClient] Readiness check error: {e}")
            return False
    
    def infer(self, text: str, audio_bytes: bytes, sample_rate: int, 
              request_id: Optional[str] = None) -> Tuple[bool, Optional[int], Optional[float], Optional[str]]:
        """
        Send single inference request over HTTP.
        Server-side dynamic batching will combine multiple concurrent requests.
        
        Returns:
            (success, token_id, pron_score, error)
        """
        try:
            self._ensure_connected()
            
            if request_id is None:
                request_id = str(uuid.uuid4())
            
            # Single sample - model has max_batch_size > 0, so include batch dimension
            # TEXT_CONTENT: [B, 1]
            text_bytes_encoded = text.encode("utf-8")
            text_np = np.array([[text_bytes_encoded]], dtype=np.object_)
            # AUDIO_DATA: [B, audio_length] - variable length raw bytes (ragged allowed)
            audio_np = np.frombuffer(audio_bytes, dtype=np.uint8).reshape(1, -1)
            # SAMPLE_RATE: [B, 1]
            sample_rate_np = np.array([[sample_rate]], dtype=np.int32)
            
            inputs = [
                self._httpclient.InferInput("TEXT_CONTENT", list(text_np.shape), "BYTES"),
                self._httpclient.InferInput("AUDIO_DATA", list(audio_np.shape), "UINT8"),
                self._httpclient.InferInput("SAMPLE_RATE", list(sample_rate_np.shape), "INT32"),
            ]
            inputs[0].set_data_from_numpy(text_np, binary_data=True)
            inputs[1].set_data_from_numpy(audio_np, binary_data=True)
            inputs[2].set_data_from_numpy(sample_rate_np, binary_data=True)
            
            outputs = [
                self._httpclient.InferRequestedOutput("OUTPUT_TOKEN_ID", binary_data=True),
                self._httpclient.InferRequestedOutput("PRON_SCORE", binary_data=True),
            ]
            
            result = self._client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs, request_id=request_id)
            
            token_id = int(result.as_numpy("OUTPUT_TOKEN_ID").flatten()[0])
            pron_score = float(result.as_numpy("PRON_SCORE").flatten()[0])
            
            return True, token_id, pron_score, None
            
        except Exception as e:
            if "Socket" in str(e) or "UNAVAILABLE" in str(e):
                self._client = None
            return False, None, None, str(e)


# =============================================================================
# Worker Process Functions (for multiprocessing with concurrent requests)
# =============================================================================

_worker_client: Optional[TritonHTTPInferenceClient] = None
_worker_url: str = ""
_worker_model: str = ""
_worker_concurrency: int = 5
_thread_local = threading.local()

def init_worker(url: str, model_name: str, concurrency: int):
    """Initialize worker process - store config, lazy connect later"""
    global _worker_url, _worker_model, _worker_client, _worker_concurrency
    _worker_url = url
    _worker_model = model_name
    _worker_client = None
    _worker_concurrency = concurrency


def _single_infer(args: Tuple[int, str, bytes, int]) -> Dict:
    """Single inference call"""
    global _worker_client, _worker_url, _worker_model
    
    if _worker_client is None:
        _worker_client = TritonHTTPInferenceClient(_worker_url, _worker_model)
    if not hasattr(_thread_local, "client") or _thread_local.client is None:
        _thread_local.client = TritonHTTPInferenceClient(_worker_url, _worker_model)
    
    idx, key, text, audio_bytes, sample_rate = args
    req_uuid = str(uuid.uuid4())
    
    start_time = time.perf_counter()
    success, token_id, pron_score, error = _thread_local.client.infer(
        text, audio_bytes, sample_rate, request_id=req_uuid
    )
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return {
        "idx": idx,
        "key": key,
        "success": success,
        "latency_ms": latency_ms,
        "token_id": token_id,
        "pron_score": pron_score,
        "error": error
    }


def worker_infer_batch_task(args: Tuple[int, List[Tuple[str, bytes, int]]]) -> Dict:
    """
    Worker task: send multiple concurrent requests to trigger server-side batching.
    Uses ThreadPoolExecutor within each worker process.
    """
    global _worker_client, _worker_url, _worker_model, _worker_concurrency
    
    if _worker_client is None:
        _worker_client = TritonHTTPInferenceClient(_worker_url, _worker_model)
    
    batch_id, batch_data = args
    batch_size = len(batch_data)
    
    # Prepare individual request args
    request_args = [(i, item[0], item[1], item[2], item[3]) for i, item in enumerate(batch_data)]
    
    results = []
    start_time = time.perf_counter()
    
    # Send concurrent requests using threads to trigger server-side batching
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(_single_infer, arg) for arg in request_args]
        for future in as_completed(futures):
            results.append(future.result())
    
    total_latency_ms = (time.perf_counter() - start_time) * 1000
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    return {
        "batch_id": batch_id,
        "keys": [r["key"] for r in results],
        "success": len(failed) == 0,
        "latency_ms": total_latency_ms,
        "batch_size": batch_size,
        "successful_count": len(successful),
        "failed_count": len(failed),
        "individual_latencies": [r["latency_ms"] for r in results],
        "token_ids": [r["token_id"] for r in successful],
        "pron_scores": [r["pron_score"] for r in successful],
        "errors": [r["error"] for r in failed] if failed else None
    }


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """Manages benchmark execution with multi-process workers and batch support"""
    
    def __init__(self, url: str, model_name: str, num_workers: int, batch_size: int):
        self.url = url
        self.model_name = model_name
        self.num_workers = num_workers
        self.batch_size = batch_size
    
    def warmup(self, test_cases: List[TestCase], num_warmup: int = 3):
        """Warmup the server with a few concurrent requests"""
        print(f"[Benchmark] Warming up with {num_warmup} batches of {self.batch_size} concurrent requests...")
        thread_local = threading.local()
        
        def warmup_call(tc: TestCase):
            if not hasattr(thread_local, "client") or thread_local.client is None:
                thread_local.client = TritonHTTPInferenceClient(self.url, self.model_name)
            return thread_local.client.infer(tc.text, tc.audio_bytes, tc.sample_rate)
        
        for i in range(num_warmup):
            # Send concurrent warmup requests
            batch = test_cases[:self.batch_size]
            with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                futures = []
                for tc in batch:
                    futures.append(executor.submit(warmup_call, tc))
                for future in as_completed(futures):
                    success, _, _, error = future.result()
                    if not success:
                        print(f"[Benchmark] Warmup request failed: {error}")
        
        print("[Benchmark] Warmup complete")
    
    def run(self, test_cases: List[TestCase], num_requests: Optional[int] = None) -> BenchmarkStats:
        """
        Run benchmark with multi-process workers and batch inference.
        
        Args:
            test_cases: List of test cases to use
            num_requests: Total number of batch requests (each contains batch_size samples)
        """
        if num_requests is None:
            num_requests = len(test_cases) // self.batch_size
        
        # Prepare batch tasks
        tasks = []
        for i in range(num_requests):
            batch_data = []
            for j in range(self.batch_size):
                tc = test_cases[(i * self.batch_size + j) % len(test_cases)]
                batch_data.append((tc.key, tc.text, tc.audio_bytes, tc.sample_rate))
            tasks.append((i, batch_data))
        
        total_samples = num_requests * self.batch_size
        
        print(f"\n{'='*60}")
        print(f"[Benchmark] Starting batch benchmark")
        print(f"  Workers: {self.num_workers}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Total batch requests: {num_requests}")
        print(f"  Total samples: {total_samples}")
        print(f"  Test cases: {len(test_cases)}")
        print(f"{'='*60}")
        
        results = []
        start_time = time.perf_counter()
        
        with Pool(
            processes=self.num_workers,
            initializer=init_worker,
            initargs=(self.url, self.model_name, self.batch_size)
        ) as pool:
            for i, result in enumerate(pool.imap_unordered(worker_infer_batch_task, tasks, chunksize=1)):
                results.append(result)
                completed_samples = (i + 1) * self.batch_size
                if (i + 1) % 10 == 0 or (i + 1) == num_requests:
                    elapsed = time.perf_counter() - start_time
                    current_qps = (i + 1) / elapsed if elapsed > 0 else 0
                    current_sps = completed_samples / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {i+1}/{num_requests} batches, {completed_samples}/{total_samples} samples ({current_qps:.2f} batch/s, {current_sps:.2f} samples/s)", end='\r')
        
        total_time = time.perf_counter() - start_time
        print()
        
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        latencies = [r["latency_ms"] for r in successful]
        
        if failed:
            print(f"\n[Benchmark] {len(failed)} batch requests failed. Sample errors:")
            error_counts = {}
            for r in failed:
                err = r.get("error", "Unknown error")
                err_short = err[:200] if isinstance(err, str) and len(err) > 200 else err
                error_counts[err_short] = error_counts.get(err_short, 0) + 1
            for err, count in sorted(error_counts.items(), key=lambda x: -x[1])[:5]:
                print(f"  [{count}x] {err}")
        
        with open('/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v3.3/infer_model/infer_res_client_batch_v2/next-2300-pa.res', 'w', encoding='utf-8') as f:
            for r in successful:
                for key, score in zip(r["keys"], r["pron_scores"]):
                    f.write(f"{key}\t{score}\n")
    
        return BenchmarkStats(
            total_requests=len(results),
            total_samples=total_samples,
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_time_sec=total_time,
            qps=len(results) / total_time if total_time > 0 else 0,
            samples_per_sec=total_samples / total_time if total_time > 0 else 0,
            latencies_ms=latencies
        )


# =============================================================================
# Statistics Reporter
# =============================================================================

class StatsReporter:
    """Formats and prints benchmark statistics"""
    
    @staticmethod
    def print_stats(stats: BenchmarkStats, batch_size: int):
        """Print formatted benchmark statistics"""
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Batch Size       : {batch_size}")
        print(f"Total Requests   : {stats.total_requests} batches")
        print(f"Total Samples    : {stats.total_samples} samples")
        print(f"Successful       : {stats.successful_requests} batches")
        print(f"Failed           : {stats.failed_requests} batches")
        print(f"Success Rate     : {stats.successful_requests/stats.total_requests*100:.2f}%")
        print(f"{'─'*60}")
        print(f"Total Time       : {stats.total_time_sec:.2f} s")
        print(f"Batch Throughput : {stats.qps:.2f} batch/s")
        print(f"Sample Throughput: {stats.samples_per_sec:.2f} samples/s")
        print(f"{'─'*60}")
        print("Latency per Batch (ms):")
        print(f"  Min    : {stats.min_latency:.2f}")
        print(f"  Avg    : {stats.avg_latency:.2f}")
        print(f"  P50    : {stats.p50_latency:.2f}")
        print(f"  P90    : {stats.p90_latency:.2f}")
        print(f"  P95    : {stats.p95_latency:.2f}")
        print(f"  P99    : {stats.p99_latency:.2f}")
        print(f"  Max    : {stats.max_latency:.2f}")
        print(f"{'─'*60}")
        print("Latency per Sample (ms):")
        print(f"  Avg    : {stats.avg_latency / batch_size:.2f}")
        print(f"  P50    : {stats.p50_latency / batch_size:.2f}")
        print(f"  P90    : {stats.p90_latency / batch_size:.2f}")
        print(f"  P99    : {stats.p99_latency / batch_size:.2f}")
        print(f"{'='*60}")
    
    @staticmethod
    def save_to_json(stats: BenchmarkStats, batch_size: int, filepath: str):
        """Save statistics to JSON file"""
        data = {
            "batch_size": batch_size,
            "total_requests": stats.total_requests,
            "total_samples": stats.total_samples,
            "successful_requests": stats.successful_requests,
            "failed_requests": stats.failed_requests,
            "total_time_sec": stats.total_time_sec,
            "batch_qps": stats.qps,
            "samples_per_sec": stats.samples_per_sec,
            "latency_per_batch_ms": {
                "min": stats.min_latency,
                "avg": stats.avg_latency,
                "p50": stats.p50_latency,
                "p90": stats.p90_latency,
                "p95": stats.p95_latency,
                "p99": stats.p99_latency,
                "max": stats.max_latency,
            },
            "latency_per_sample_ms": {
                "avg": stats.avg_latency / batch_size,
                "p50": stats.p50_latency / batch_size,
                "p90": stats.p90_latency / batch_size,
                "p99": stats.p99_latency / batch_size,
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[StatsReporter] Results saved to {filepath}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="High-Performance HTTP Benchmark Client for Kimi Audio Triton Service (Batch Version)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--url", type=str, default="localhost:8010",
                        help="Triton HTTP server URL (host:port)")
    parser.add_argument("--model", type=str, default="kimi_whole",
                        help="Model name to benchmark")
    
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Batch size for each inference request")
    parser.add_argument("--num_requests", type=int, default=100,
                        help="Total number of batch inference requests")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of worker processes (parallel clients)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup batch requests before benchmark")
    
    parser.add_argument("--label_file", type=str,
                        default="/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/api_data/next/tal-k12/test/label_sent_score",
                        help="Path to label file (key\\ttext format)")
    parser.add_argument("--wavpath_file", type=str,
                        default="/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/api_data/next/tal-k12/wavpath_merged",
                        help="Path to wavpath file (key\\tpath format)")
    parser.add_argument("--max_data", type=int, default=None,
                        help="Maximum number of test cases to load")
    
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save results as JSON")
    
    args = parser.parse_args()
    
    # Validate batch size
    if args.batch_size < 1 or args.batch_size > 5:
        print(f"[Error] Batch size must be between 1 and 5 (got {args.batch_size})")
        return 1
    
    test_cases = DataLoader.load_dataset(
        args.label_file, args.wavpath_file, max_items=args.max_data
    )
    
    if not test_cases:
        print("[Error] No valid test cases loaded")
        return 1
    
    if len(test_cases) < args.batch_size:
        print(f"[Error] Not enough test cases ({len(test_cases)}) for batch size {args.batch_size}")
        return 1
    
    print(f"\n[Main] Connecting to Triton HTTP server at {args.url}")
    client = TritonHTTPInferenceClient(args.url, args.model)
    if not client.is_server_ready():
        print(f"[Error] Server not ready or model '{args.model}' not loaded")
        return 1
    print("[Main] Server is ready")
    
    runner = BenchmarkRunner(args.url, args.model, args.num_workers, args.batch_size)
    
    if args.warmup > 0:
        runner.warmup(test_cases[:args.batch_size], args.warmup)
    
    stats = runner.run(test_cases, args.num_requests)
    
    StatsReporter.print_stats(stats, args.batch_size)
    
    if args.output_json:
        StatsReporter.save_to_json(stats, args.batch_size, args.output_json)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
