// Presentation helpers shared across components. No API or React here.

// Short axis-friendly labels for the full engineered feature set (see
// models/v*/feature_names.json; v2 omits age and the age-derived names, so a
// given version may use a subset). Age bands are handled by prefix below.
export const FEATURE_LABELS = {
  RevolvingUtilizationOfUnsecuredLines: "Revolving credit utilization",
  age: "Age",
  "NumberOfTime30-59DaysPastDueNotWorse": "Times 30–59 days past due",
  DebtRatio: "Debt-to-income ratio",
  MonthlyIncome: "Monthly income",
  NumberOfOpenCreditLinesAndLoans: "Open credit lines & loans",
  NumberOfTimes90DaysLate: "Times 90+ days late",
  NumberRealEstateLoansOrLines: "Real-estate loans / lines",
  "NumberOfTime60-89DaysPastDueNotWorse": "Times 60–89 days past due",
  NumberOfDependents: "Number of dependents",
  income_missing: "Income not provided",
  dependents_missing: "Dependents not provided",
  total_past_due_count: "Total past-due periods",
  has_past_due: "Any past-due history",
  income_per_dependent: "Income per dependent",
  has_dependents: "Has dependents",
  credit_lines_per_year_of_age: "Credit lines per year of age",
};

export function featureLabel(name) {
  if (FEATURE_LABELS[name]) return FEATURE_LABELS[name];
  if (name.startsWith("age_bin_")) return `Age band ${name.slice("age_bin_".length)}`;
  return name;
}

export function pct(x, digits = 1) {
  if (x == null || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

// Signed, with a real minus sign for typography.
export function signed(x, digits = 2) {
  if (x == null || Number.isNaN(x)) return "—";
  const sign = x >= 0 ? "+" : "−";
  return `${sign}${Math.abs(x).toFixed(digits)}`;
}

export function fmtValue(v) {
  if (v === null || v === undefined || v === "") return "—";
  const n = Number(v);
  if (Number.isNaN(n)) return String(v);
  if (Number.isInteger(n)) return n.toLocaleString();
  return n.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

export function fmtDateTime(iso) {
  try {
    return new Date(iso).toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    });
  } catch {
    return iso;
  }
}

export const DECISION_META = {
  approved: { label: "Approved", tone: "good" },
  referred: { label: "Referred", tone: "warning" },
  denied: { label: "Denied", tone: "critical" },
};

export function decisionMeta(decision) {
  return DECISION_META[decision] || { label: decision, tone: "muted" };
}
