# 统计各分位数下的音频时长及RTF
import numpy as np

def calc_rtf(duration_file, max_wait_ms=150, quantiles=[0.5, 0.9, 0.95, 0.99]):
	# 读取时长数据
	with open(duration_file, 'r') as f:
		durations = [float(line.strip()) for line in f if line.strip()]
	durations = np.array(durations)
	print(f"总样本数: {len(durations)}")
	print(f"最大等待时长: {max_wait_ms} ms")
	print("分位数\t音频时长(ms)\tRTF")
	for q in quantiles:
		dur = np.percentile(durations, q*100)
		# 要保证处理时间 <= max_wait_ms，需要满足: processing_time = dur * RTF <= max_wait_ms
		# 因此允许的最大 RTF = max_wait_ms / dur
		if dur == 0:
			rtf = float('inf')
		else:
			rtf = max_wait_ms / dur
		print(f"{int(q*100)}%\t{dur:.2f}\t\t{rtf:.4f}")

if __name__ == "__main__":
	import sys
	if len(sys.argv) < 2:
		print("用法: python shangxain_rtf.py <duration_file> [max_wait_ms]")
		exit(1)
	duration_file = sys.argv[1]
	max_wait_ms = int(sys.argv[2]) if len(sys.argv) > 2 else 150
	calc_rtf(duration_file, max_wait_ms)
