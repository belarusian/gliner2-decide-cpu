from gliner2 import AutoExtractor
from gliner2.classification import Classifier, ClassificationSchema
from gliner2.classification import constraints as C

model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide", map_location="cpu")

print("=== 1. Single-label intent routing ===")
result = model.classify_text(
    "My subscription renewed on April 15 for ¥5,400 after the service was already down. Can I get that charge refunded?",
    {"intent": [
        "order_status", "refund_request", "cancel_subscription", "update_payment",
        "login_problem", "shipping_delay", "bug_report", "speak_to_human", "other",
    ]},
    include_confidence=True,
)
print(result)

print("\n=== 2. Several decisions in one pass ===")
result = model.classify_text(
    "From: compliance@group.example\nSubject: Protocol update - action required today\n\n"
    "Please confirm the new retention rule is applied before Friday's audit.",
    {
        "intent": ["fyi", "request", "approval", "complaint", "newsletter", "security_alert"],
        "urgency": ["low", "normal", "high", "critical"],
        "route": ["support", "billing", "legal", "security", "finance", "archive"],
    },
)
print(result)

print("\n=== 3. Multi-label (aspects) ===")
result = model.classify_text(
    "Battery dies before lunch, but the keyboard and the screen are the best I have used on a laptop.",
    {"aspects": {"labels": ["battery", "keyboard", "screen", "camera", "price", "support"],
                  "multi_label": True, "cls_threshold": 0.4}},
)
print(result)

print("\n=== 4. Constrained decoding (cross-task rules) ===")
clf = Classifier.from_pretrained("fastino/GLiNER2.5-Decide", map_location="cpu")
schema = (
    ClassificationSchema()
    .single("intent", ["read", "write", "delete"])
    .multi("effects", ["read_only", "create", "modify", "delete"], min_labels=1, max_labels=2)
    .constrain(
        C.implies(("intent", "delete"), ("effects", "delete")),
        C.excludes(("intent", "read"), ("effects", "delete")),
    )
)
result = clf.classify("Delete the temporary file from /tmp", schema)
print(f"intent={result.value('intent')} effects={list(result.selected('effects'))} feasible={result.feasible}")
