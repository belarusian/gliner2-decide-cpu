import time
from gliner2 import AutoExtractor

LABELS = ["refund", "shipping_delay", "bug_report", "other"]
TEXT = "My package never arrived and I want my money back"
N = 20

model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide", map_location="cpu")

for _ in range(3):
    model.classify_text(TEXT, {"intent": LABELS})

latencies = []
for _ in range(N):
    t = time.perf_counter()
    model.classify_text(TEXT, {"intent": LABELS})
    latencies.append((time.perf_counter() - t) * 1000)

latencies.sort()
print(f"n={N} warm requests on CPU")
print(f"mean {sum(latencies)/N:.1f} ms | p50 {latencies[N//2]:.1f} ms | min {latencies[0]:.1f} ms | max {latencies[-1]:.1f} ms")
