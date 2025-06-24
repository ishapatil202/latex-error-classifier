# predict_latex_error.py  (robust semantic-equivalence version)

import json
import os
import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sympy import sympify, Eq
from sympy.parsing.latex import parse_latex

# ---------------- 1.  Load tokenizer & label-encoder ----------------
MODEL_DIR = "bert_latex_classifier"          # folder with saved weights
tokenizer  = AutoTokenizer.from_pretrained(MODEL_DIR)

# rebuild LabelEncoder from the SAME training JSONL
with open("latex_error_dataset_with_no_error.jsonl", "r") as f:
    all_labels = [json.loads(line)["label"] for line in f]
label_encoder = LabelEncoder().fit(all_labels)
NUM_LABELS    = len(label_encoder.classes_)

# ---------------- 2.  Re-create two-layer MathBERT model ------------
class MathBERTTwoLayerClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.30)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids=None, attention_mask=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.pooler_output)
        logits = self.classifier(pooled)
        return {"logits": logits}

model = MathBERTTwoLayerClassifier("tbs17/MathBERT", NUM_LABELS)
state_dict_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
model.eval()

# ---------------- 3.  Robust semantic-equivalence check -------------
def answers_equivalent(lx1: str, lx2: str) -> bool:
    """
    Try to decide if two answers are mathematically identical.
    1)  parse LaTeX with sympy.parse_latex
    2)  fall back to sympy.sympify for plain numbers
    3)  on any error → return False (so classifier runs)
    """
    def _to_expr(lx: str):
        try:
            # try LaTeX parsing first
            return parse_latex(lx)
        except Exception:
            try:
                # try plain numeric / rational
                return sympify(lx)
            except Exception:
                return None

    expr1, expr2 = _to_expr(lx1), _to_expr(lx2)
    if expr1 is None or expr2 is None:
        # couldn’t parse → treat as NOT equivalent
        return False
    try:
        return bool(Eq(expr1, expr2))
    except Exception:
        return False

# ---------------- 4.  Main prediction helper -----------------------
def predict_error(problem_lx: str, correct_lx: str, student_lx: str) -> str:
    # 4-A  Semantic guard
    if answers_equivalent(student_lx, correct_lx):
        return "no_error"

    # 4-B  Otherwise use classifier on (problem + student answer)
    text = f"Problem: {problem_lx} || Student Answer: <ANS>{student_lx}</ANS>"
    toks = tokenizer(text,
                     return_tensors="pt",
                     padding="max_length",
                     truncation=True,
                     max_length=128)

    toks.pop("token_type_ids", None)          # our model ignores this field
    with torch.no_grad():
        logits  = model(**toks)["logits"]
        pred_id = int(torch.argmax(logits, dim=1))
        return label_encoder.inverse_transform([pred_id])[0]

# ---------------- 5.  Quick sanity-check ---------------------------
if __name__ == "__main__":
    test_cases = [
        # should be “no_error”
        {
            "problem":  "$\\frac{2}{6} \\div \\frac{1}{5}$",
            "correct":  "$1 \\frac{2}{3}$",
            "student":  "$1 \\frac{2}{3}$",
        },
        # should be some error label
        {
            "problem":  "$\\frac{2}{6} \\div \\frac{1}{5}$",
            "correct":  "$1 \\frac{2}{3}$",
            "student":  "$\\frac{2}{6} \\times \\frac{1}{5}$",
        },
        {
            "problem":  "$2 \\frac{1}{2} + 3 \\frac{2}{5}$",
            "correct":  "$5 \\frac{9}{10}$",
            "student":  "$5 \\frac{1}{6}$",
        },
    ]

    print("\nPredictions:\n")
    for case in test_cases:
        lbl = predict_error(case["problem"], case["correct"], case["student"])
        print(f" Problem:  {case['problem']}")
        print(f" Correct:  {case['correct']}")
        print(f" Student:  {case['student']}")
        print(f" ➜ Predicted label: {lbl}\n")
