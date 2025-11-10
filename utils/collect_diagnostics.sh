#!/bin/bash
OUTDIR=/mnt/pfs_l2/jieti_team/SFT/hupeng/github/Kimi-Audio-batch/logs/nccl_diag_$(date +%Y%m%d_%H%M%S)
mkdir -p $OUTDIR
echo "Saving diagnostics to $OUTDIR"

# 1) basic host info
hostname > $OUTDIR/hostname.txt
uname -a > $OUTDIR/uname.txt
nvidia-smi -q > $OUTDIR/nvidia_smi_q.txt
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv > $OUTDIR/nvidia_smi_apps.csv


# 5) dmesg and journal
dmesg | tail -n 500 > $OUTDIR/dmesg_tail.txt
journalctl --since "10 minutes ago" > $OUTDIR/journal_recent.txt

# 6) python stack traces using gdb (requires root or same user)
# Replace <PID> with actual PIDs from ps.txt if needed - here we try all python PIDs
for pid in $(awk '{print $2}' $OUTDIR/ps.txt | grep -E '^[0-9]+$' || true); do
  echo "=== PID $pid ===" > $OUTDIR/gdb_backtrace_${pid}.txt
  if [ -d /proc/$pid ]; then
    # gdb backtrace (native + python)
    gdb -p $pid --batch -ex "thread apply all bt" &> $OUTDIR/gdb_backtrace_${pid}.txt || true
  fi
done

# 7) Python-level stacks using faulthandler: send SIGUSR1 to python procs (if process enabled)
for pid in $(awk '{print $2}' $OUTDIR/ps.txt | grep -E '^[0-9]+$' || true); do
  kill -SIGUSR1 $pid 2>/dev/null || true
done