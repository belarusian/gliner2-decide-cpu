# GLiNER2.5-Decide on Apple Silicon (CPU-only)

Local, zero-cost structured decision-making with [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) — a 340M **encoder-based** decision model that classifies text against any label set in a single forward pass. No tokens generated, nothing parsed, no API keys.

This repo documents getting it running on a MacBook Air and measures the latency.

## What it is (in one paragraph)

GLiNER2.5-Decide is **not** a chatbot. It's a DeBERTa-v3-large encoder + a tiny 2-layer MLP head. You pass text and a set of candidate labels; it returns the matching label with a confidence score. A single forward pass, ~340M params, runs on CPU. On `fastino/fast-decisions` it scores **60.2%** exact-match, beating a Qwen3.5-4B decoder (56.4%) at ~1/10th the size.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "gliner2[local]"
```

## Usage

```python
from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide", map_location="cpu")

model.classify_text(
    "My subscription renewed after the service was already down. Can I get a refund?",
    {"intent": ["order_status", "refund_request", "cancel_subscription", "login_problem", "other"]},
    include_confidence=True,
)
# {'intent': {'label': 'refund_request', 'confidence': 0.997}}
```

Run all four demos (intent routing, multi-head triage, multi-label aspects, constrained decoding):

```bash
.venv/bin/python example.py
```

## Results (MacBook Air, CPU only)

Example outputs from `example.py`:

| Task | Input | Output |
|------|-------|--------|
| Intent routing | "subscription renewed... can I get that charge refunded?" | `refund_request` (conf 0.997) |
| Email triage (3 heads) | compliance audit email | `intent=request`, `urgency=critical`, `route=legal` |
| Multi-label aspects | "battery dies... keyboard and screen are best" | `[battery, keyboard, screen]` |
| Constrained decode | "Delete the temporary file" | `intent=delete` → `effects=[delete]`, feasible=True |

### Timing

`bench.py` runs 20 warm requests, single label set of 4:

| Hardware | Per request |
|----------|-------------|
| MacBook Air (Apple Silicon, CPU) | **p50 102.4 ms, mean 105.3 ms** |
| Intel Xeon 8581C, 48 vCPU (vendor) | 167.3 ms (vendor) |
| NVIDIA V100 (vendor) | 38.3 ms |

CPU-only inference is practical — the model is small enough that it beats the vendor's own Xeon baseline on a laptop.

## Constrained decoding

The interesting bit: rules between labels are enforced by a **combinatorial search on top of the encoder's scores** — not by the model itself. The neural net just produces per-label logits; a solver picks the highest-scoring legal combination.

```python
from gliner2.classification import Classifier, ClassificationSchema
from gliner2.classification import constraints as C

clf = Classifier.from_pretrained("fastino/GLiNER2.5-Decide")
schema = (
    ClassificationSchema()
    .single("intent", ["read", "write", "delete"])
    .multi("effects", ["read_only", "create", "modify", "delete"], min_labels=1)
    .constrain(
        C.implies(("intent", "delete"), ("effects", "delete")),
        C.excludes(("intent", "read"), ("effects", "delete")),
    )
)
result = clf.classify("Delete the temporary file from /tmp", schema)
# intent=delete, effects=[delete]  — rule satisfied without the model "knowing" the rule
```

## The head, literally

The whole classification head is two linear layers (2.1M params, 8.4 MB):

```python
nn.Sequential(
    nn.Linear(1024, 2048),  # from each [L] marker's hidden state
    nn.ReLU(),
    nn.Dropout(0.1),         # no-op at inference
    nn.Linear(2048, 1),      # → one logit per label
)
```

Applied to the encoder hidden state at each `[L]` label marker. The encoder does the understanding; this head just scores compatibility.

## Files

- `example.py` — four worked demos
- `bench.py` — timing benchmark

## License

Model weights Apache 2.0 (Fastino Labs). This repo's example code: MIT.
