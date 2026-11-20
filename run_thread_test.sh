#!/bin/bash
# Thread test for cv2.cvtColor() (Reviewer 2, Comment 3). Run after: sudo jetson_clocks
LOG=thread_test_$(date +%Y%m%d_%H%M).log
{
echo "== 1. Default (reproduce Table 3: about 1.5 vs 0.37 ms)"
python3 thread_test.py --pipe cpu
python3 thread_test.py --pipe gpu
echo "== 2. OpenCV single thread"
python3 thread_test.py --pipe cpu --cv 1
python3 thread_test.py --pipe gpu --cv 1
echo "== 3. OpenCV 4 threads"
python3 thread_test.py --pipe cpu --cv 4
python3 thread_test.py --pipe gpu --cv 4
echo "== 4. PyTorch CPU single thread"
python3 thread_test.py --pipe cpu --torch 1
echo "== 5. OpenMP passive wait (idle threads sleep instead of spinning)"
OMP_WAIT_POLICY=PASSIVE python3 thread_test.py --pipe cpu
} 2>&1 | tee "$LOG"
echo; echo "== Summary"; grep RESULT "$LOG"
