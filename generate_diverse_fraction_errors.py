import random
import pandas as pd
from fractions import Fraction

# --- Helper Functions ---
def random_fraction():
    den = random.randint(2, 20)
    num = random.randint(1, den-1)
    return num, den

def random_mixed():
    whole = random.randint(1, 5)
    num, den = random_fraction()
    return whole, num, den

def to_improper(whole, num, den):
    return whole * den + num, den

def format_fraction(num, den):
    if den == 0:
        return "Undefined"
    f = Fraction(num, den)
    num, den = f.numerator, f.denominator
    if abs(num) >= den and den != 1:
        whole = abs(num) // den
        rem = abs(num) % den
        sign = '-' if num < 0 else ''
        if rem == 0:
            return f"{sign}{whole}"
        else:
            return f"{sign}{whole} {rem}/{den}"
    else:
        return f"{num}/{den}"

def random_operand():
    typ = random.choice(['mixed', 'proper', 'improper'])
    if typ == 'mixed':
        w, n, d = random_mixed()
        return typ, (w, n, d)
    elif typ == 'proper':
        n, d = random_fraction()
        return typ, (n, d)
    else:
        d = random.randint(2, 20)
        n = random.randint(d, 2*d)
        return typ, (n, d)

def operand_to_improper(typ, value):
    if typ == 'mixed':
        w, n, d = value
        return w * d + n, d
    else:
        n, d = value
        return n, d

def operand_to_string(typ, value):
    if typ == 'mixed':
        w, n, d = value
        return f"{w} {n}/{d}"
    else:
        n, d = value
        return f"{n}/{d}"

# --- Error Generation Functions ---
def conv_a_error(n1, d1, n2, d2):
    return format_fraction(n1 + n2, d1 + d2)

def conv_b_error(n1, d1, n2, d2):
    choice = random.choice([
        lambda: format_fraction(n1 * n2, d1 * d2),
        lambda: format_fraction(n1 * d2, d1 * n2),
        lambda: format_fraction(n2, d1),
        lambda: format_fraction(n1, d2),
    ])
    return choice()

def conv_c_error(n1, d1, n2, d2):
    return format_fraction(n1 + n2, d1)

def conv_d_error(n1, d1, n2, d2):
    return format_fraction(n1 - n2, d1 + d2)

def po_a_error(correct_num, correct_den):
    return format_fraction(-correct_num, correct_den)

def po_b_error(correct_num, correct_den):
    if correct_num != 0:
        return format_fraction(correct_den, correct_num)
    else:
        return "Undefined"

def po_c_error(n1, d1, n2, d2):
    return format_fraction(n1 * n2, d1 + d2)

def po_d_error(n1, d1, n2, d2):
    den = d1 - d2 if d1 != d2 else 1
    return format_fraction(n1 + n2, den)

def pf_a_error(n1, d1, n2, d2):
    return format_fraction(n1, d1 + d2)

def pf_b_error(n1, d1, n2, d2):
    return format_fraction(n1 + n2, d2)

def pf_c_error(n1, d1, n2, d2):
    return format_fraction(n1 * n2, d1)

def pf_d_error(n1, d1, n2, d2):
    return format_fraction(n1 + n2, d2)

def pf_e_error(n1, d1, n2, d2):
    return format_fraction(n1 - n2, d1 * d2)

def a_error(num, den):
    return format_fraction(num + 1, den)

# --- Main Data Generation ---
data = []
labels = [
    ("conv-a", conv_a_error),
    ("conv-b", conv_b_error),
    ("conv-c", conv_c_error),
    ("conv-d", conv_d_error),
    ("po-a", po_a_error),
    ("po-b", po_b_error),
    ("po-c", po_c_error),
    ("po-d", po_d_error),
    ("pf-a", pf_a_error),
    ("pf-b", pf_b_error),
    ("pf-c", pf_c_error),
    ("pf-d", pf_d_error),
    ("pf-e", pf_e_error),
    ("a", a_error),
]

unique_problems = set()
target_count = 30000

while len(data) < target_count:
    op1_type, op1_value = random_operand()
    op2_type, op2_value = random_operand()
    op = random.choice(['+', '-', 'x', '÷'])

    op1_str = operand_to_string(op1_type, op1_value)
    op2_str = operand_to_string(op2_type, op2_value)
    problem = f"{op1_str} {op} {op2_str}"

    n1, d1 = operand_to_improper(op1_type, op1_value)
    n2, d2 = operand_to_improper(op2_type, op2_value)

    if op == '+':
        num = n1 * d2 + n2 * d1
        den = d1 * d2
    elif op == '-':
        num = n1 * d2 - n2 * d1
        den = d1 * d2
    elif op == 'x':
        num = n1 * n2
        den = d1 * d2
    else:
        num = n1 * d2
        den = d1 * n2 if n2 != 0 else 1

    key = (problem, n1, d1, n2, d2, op)
    if key in unique_problems:
        continue
    unique_problems.add(key)

    correct_frac = Fraction(num, den)
    correct_num = correct_frac.numerator
    correct_den = correct_frac.denominator

    for label, err_func in labels:
        if label in ["po-a", "po-b", "a"]:
            student_ans = err_func(correct_num, correct_den)
        else:
            student_ans = err_func(n1, d1, n2, d2)
        row = {
            "input": f"Problem: {problem} | Student Answer: {student_ans}",
            "label": label
        }
        data.append(row)
        if len(data) >= target_count:
            break

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("diversed-data.csv", index=False)
print("Generated 2000+ diverse, label-specific fraction problems in 'diversed-data.csv'") 