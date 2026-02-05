#!/usr/bin/env python3
"""
High-Performance HTTP Benchmark Client for Kimi Audio Triton Service
Features:
- Multi-process architecture to bypass Python GIL
- Modular design with separate data loading, inference, and statistics
- Support for sync benchmarking over Triton HTTP endpoint
- Detailed latency statistics (QPS, P50, P90, P99, etc.)
"""

import argparse
import os
import sys
import time
import wave
import json
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import multiprocessing as mp
from multiprocessing import Pool

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
    successful_requests: int
    failed_requests: int
    total_time_sec: float
    qps: float
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
# Triton HTTP Client Module
# =============================================================================

class TritonHTTPInferenceClient:
    """Handles Triton HTTP inference requests"""
    
    def __init__(self, url: str, model_name: str = "kimi_ensemble"):
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
    
    def infer(self, text: str, audio_bytes: bytes, sample_rate: int, request_id: Optional[str] = None) -> Tuple[bool, Optional[int], Optional[float], Optional[str]]:
        """
        Send inference request over HTTP and return (success, token_id, pron_score, error)
        No batch dimension (max_batch_size=0)
        """
        try:
            self._ensure_connected()
            
            # Generate request ID if not provided
            if request_id is None:
                request_id = str(uuid.uuid4())
            
            # No batch dimension - shapes: [1] for text/sample_rate, [audio_len] for audio
            text_np = np.array([text.encode("utf-8")], dtype=np.object_)
            audio_np = np.frombuffer(audio_bytes, dtype=np.uint8)
            sample_rate_np = np.array([sample_rate], dtype=np.int32)
            
            inputs = [
                self._httpclient.InferInput("TEXT_CONTENT", text_np.shape, self._np_to_triton_dtype(text_np.dtype)),
                self._httpclient.InferInput("AUDIO_DATA", audio_np.shape, "UINT8"),
                self._httpclient.InferInput("SAMPLE_RATE", sample_rate_np.shape, "INT32"),
            ]
            # Use binary_data for efficiency with HTTP
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
# Worker Process Functions (for multiprocessing)
# =============================================================================

_worker_client: Optional[TritonHTTPInferenceClient] = None
_worker_url: str = ""
_worker_model: str = ""

def init_worker(url: str, model_name: str):
    """Initialize worker process - store config, lazy connect later"""
    global _worker_url, _worker_model, _worker_client
    _worker_url = url
    _worker_model = model_name
    _worker_client = None


def worker_infer_task(args: Tuple[int, str, bytes, int]) -> Dict:
    """
    Worker task function for multiprocessing.
    Args: (request_id, text, audio_bytes, sample_rate)
    Returns: dict with result info
    """
    global _worker_client, _worker_url, _worker_model
    
    if _worker_client is None:
        _worker_client = TritonHTTPInferenceClient(_worker_url, _worker_model)
    
    request_id, text, audio_bytes, sample_rate = args
    
    # Generate UUID for request
    req_uuid = str(uuid.uuid4())
    
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        start_time = time.perf_counter()
        success, token_id, pron_score, error = _worker_client.infer(text, audio_bytes, sample_rate, request_id=req_uuid)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        if success:
            return {
                "request_id": request_id,
                "success": True,
                "latency_ms": latency_ms,
                "token_id": token_id,
                "pron_score": pron_score,
                "error": None
            }
        
        last_error = error
        if "Socket" not in str(error) and "UNAVAILABLE" not in str(error):
            break
        time.sleep(0.1 * (attempt + 1))
    
    return {
        "request_id": request_id,
        "success": False,
        "latency_ms": latency_ms,
        "token_id": None,
        "pron_score": None,
        "error": last_error
    }


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """Manages benchmark execution with multi-process workers"""
    
    def __init__(self, url: str, model_name: str, num_workers: int):
        self.url = url
        self.model_name = model_name
        self.num_workers = num_workers
    
    def warmup(self, test_case: TestCase, num_warmup: int = 3):
        """Warmup the server with a few requests"""
        print(f"[Benchmark] Warming up with {num_warmup} requests...")
        client = TritonHTTPInferenceClient(self.url, self.model_name)
        
        for i in range(num_warmup):
            success, _, _, error = client.infer(
                test_case.text, test_case.audio_bytes, test_case.sample_rate
            )
            if not success:
                print(f"[Benchmark] Warmup request {i+1} failed: {error}")
        
        print("[Benchmark] Warmup complete")
    
    def run(self, test_cases: List[TestCase], num_requests: Optional[int] = None) -> BenchmarkStats:
        """
        Run benchmark with multi-process workers.
        
        Args:
            test_cases: List of test cases to use
            num_requests: Total number of requests (cycles through test_cases if needed)
        """
        if num_requests is None:
            num_requests = len(test_cases)
        
        tasks = []
        for i in range(num_requests):
            tc = test_cases[i % len(test_cases)]
            tasks.append((tc.key, tc.text, tc.audio_bytes, tc.sample_rate))
        
        print(f"\n{'='*60}")
        print(f"[Benchmark] Starting benchmark")
        print(f"  Workers: {self.num_workers}")
        print(f"  Total requests: {num_requests}")
        print(f"  Test cases: {len(test_cases)}")
        print(f"{'='*60}")
        
        results = []
        start_time = time.perf_counter()
        
        with Pool(
            processes=self.num_workers,
            initializer=init_worker,
            initargs=(self.url, self.model_name)
        ) as pool:
            for i, result in enumerate(pool.imap_unordered(worker_infer_task, tasks, chunksize=1)):
                results.append(result)
                if (i + 1) % 50 == 0 or (i + 1) == num_requests:
                    elapsed = time.perf_counter() - start_time
                    current_qps = (i + 1) / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {i+1}/{num_requests} ({current_qps:.2f} QPS)", end='\r')
        
        total_time = time.perf_counter() - start_time
        print()
        
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        latencies = [r["latency_ms"] for r in successful]
        
        if failed:
            print(f"\n[Benchmark] {len(failed)} requests failed. Sample errors:")
            error_counts = {}
            for r in failed:
                err = r.get("error", "Unknown error")
                err_short = err[:200] if isinstance(err, str) and len(err) > 200 else err
                error_counts[err_short] = error_counts.get(err_short, 0) + 1
            for err, count in sorted(error_counts.items(), key=lambda x: -x[1])[:5]:
                print(f"  [{count}x] {err}")

        with open('/mnt/pfs_l2/jieti_team/SFT/hupeng/github/kimi-sft/logs/kimi_client_log/next2300_benchmark_pa.txt', 'w') as f:
            for r in successful:
                f.write(f"{r['request_id']}\t{r['pron_score']}\n")

        return BenchmarkStats(
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_time_sec=total_time,
            qps=len(results) / total_time if total_time > 0 else 0,
            latencies_ms=latencies
        )


# =============================================================================
# Statistics Reporter
# =============================================================================

class StatsReporter:
    """Formats and prints benchmark statistics"""
    
    @staticmethod
    def print_stats(stats: BenchmarkStats):
        """Print formatted benchmark statistics"""
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Total Requests   : {stats.total_requests}")
        print(f"Successful       : {stats.successful_requests}")
        print(f"Failed           : {stats.failed_requests}")
        print(f"Success Rate     : {stats.successful_requests/stats.total_requests*100:.2f}%")
        print(f"{'─'*60}")
        print(f"Total Time       : {stats.total_time_sec:.2f} s")
        print(f"QPS (Throughput) : {stats.qps:.2f} req/s")
        print(f"{'─'*60}")
        print("Latency (ms):")
        print(f"  Min    : {stats.min_latency:.2f}")
        print(f"  Avg    : {stats.avg_latency:.2f}")
        print(f"  P50    : {stats.p50_latency:.2f}")
        print(f"  P90    : {stats.p90_latency:.2f}")
        print(f"  P95    : {stats.p95_latency:.2f}")
        print(f"  P99    : {stats.p99_latency:.2f}")
        print(f"  Max    : {stats.max_latency:.2f}")
        print(f"{'='*60}")
    
    @staticmethod
    def save_to_json(stats: BenchmarkStats, filepath: str):
        """Save statistics to JSON file"""
        data = {
            "total_requests": stats.total_requests,
            "successful_requests": stats.successful_requests,
            "failed_requests": stats.failed_requests,
            "total_time_sec": stats.total_time_sec,
            "qps": stats.qps,
            "latency_ms": {
                "min": stats.min_latency,
                "avg": stats.avg_latency,
                "p50": stats.p50_latency,
                "p90": stats.p90_latency,
                "p95": stats.p95_latency,
                "p99": stats.p99_latency,
                "max": stats.max_latency,
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
        description="High-Performance HTTP Benchmark Client for Kimi Audio Triton Service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--url", type=str, default="localhost:8010",
                        help="Triton HTTP server URL (host:port)")
    parser.add_argument("--model", type=str, default="kimi_ensemble",
                        help="Model name to benchmark")
    
    parser.add_argument("--num_requests", type=int, default=None,
                        help="Total number of inference requests")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of worker processes (parallel clients)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup requests before benchmark")
    
    parser.add_argument("--label_file", type=str,
                        default="/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/api_data/next/tal-k12/test/label_snt_score_merged",
                        help="Path to label file (key\\ttext format)")
    parser.add_argument("--wavpath_file", type=str,
                        default="/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/api_data/next/tal-k12/wavpath_merged",
                        help="Path to wavpath file (key\\tpath format)")
    parser.add_argument("--max_data", type=int, default=None,
                        help="Maximum number of test cases to load")
    
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save results as JSON")
    
    args = parser.parse_args()
    
    test_cases = DataLoader.load_dataset(
        args.label_file, args.wavpath_file, max_items=args.max_data
    )
    
    if not test_cases:
        print("[Error] No valid test cases loaded")
        return 1
    
    print(f"\n[Main] Connecting to Triton HTTP server at {args.url}")
    client = TritonHTTPInferenceClient(args.url, args.model)
    if not client.is_server_ready():
        print(f"[Error] Server not ready or model '{args.model}' not loaded")
        return 1
    print("[Main] Server is ready")
    
    runner = BenchmarkRunner(args.url, args.model, args.num_workers)
    
    if args.warmup > 0:
        runner.warmup(test_cases[0], args.warmup)
    
    stats = runner.run(test_cases, args.num_requests)
    
    StatsReporter.print_stats(stats)
    
    if args.output_json:
        StatsReporter.save_to_json(stats, args.output_json)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
