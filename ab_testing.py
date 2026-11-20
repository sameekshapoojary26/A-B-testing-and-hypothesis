import pandas as pd
import numpy as np
from scipy.stats import norm, chi2_contingency
from reportlab.pdfgen import canvas

# Load dataset
df = pd.read_csv("ab_test_data.csv")

# Expected columns: group and converted
df.columns = df.columns.str.lower().str.strip()

group_col = next(c for c in df.columns if c in
                 ["group", "variant", "test_group"])
conversion_col = next(c for c in df.columns if c in
                      ["converted", "conversion", "is_converted"])

control = df[df[group_col].str.lower().isin(["a", "control"])]
variant = df[df[group_col].str.lower().isin(["b", "variant"])]

n_a, n_b = len(control), len(variant)
x_a = control[conversion_col].sum()
x_b = variant[conversion_col].sum()

rate_a = x_a / n_a
rate_b = x_b / n_b
lift = rate_b - rate_a

# Hypotheses
H0 = "There is no significant difference."
H1 = "There is a significant difference."

# Two-proportion Z-test
pooled = (x_a + x_b) / (n_a + n_b)
se = np.sqrt(pooled * (1-pooled) * (1/n_a + 1/n_b))
z = lift / se
p_value = 2 * (1 - norm.cdf(abs(z)))

# 95% Confidence interval
se_ci = np.sqrt(rate_a*(1-rate_a)/n_a +
                rate_b*(1-rate_b)/n_b)
lower = lift - 1.96 * se_ci
upper = lift + 1.96 * se_ci

# Chi-square test
table = [[x_a, n_a-x_a], [x_b, n_b-x_b]]
chi2, chi_p, _, _ = chi2_contingency(table)

# Decision
decision = "REJECT H0" if p_value < 0.05 else "FAIL TO REJECT H0"

# Report
report = f"""
A/B TESTING AND HYPOTHESIS TESTING REPORT

Null Hypothesis: {H0}
Alternative Hypothesis: {H1}

Control Visitors: {n_a}
Variant Visitors: {n_b}

Control Conversions: {x_a}
Variant Conversions: {x_b}

Control Conversion Rate: {rate_a:.2%}
Variant Conversion Rate: {rate_b:.2%}

Conversion Lift: {lift:.2%}

Z Statistic: {z:.4f}
Z-test P-value: {p_value:.6f}

95% Confidence Interval:
Lower: {lower:.4%}
Upper: {upper:.4%}

Chi-square Statistic: {chi2:.4f}
Chi-square P-value: {chi_p:.6f}

Decision: {decision}

EXECUTIVE RECOMMENDATION:
""" + (
    "Launch Variant B. The improvement is statistically significant."
    if p_value < 0.05 and lift > 0
    else "Keep the control version. The variant has not demonstrated a significant positive improvement."
)

print(report)

# Generate PDF
pdf = canvas.Canvas("AB_Test_Report.pdf")
y = 800

for line in report.split("\n"):
    pdf.drawString(45, y, line[:100])
    y -= 20
    if y < 50:
        pdf.showPage()
        y = 800

pdf.save()

print("PDF report generated successfully!")
