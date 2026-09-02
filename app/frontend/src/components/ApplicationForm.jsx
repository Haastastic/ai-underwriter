import { useState } from "react";

export const FIELDS = [
  {
    name: "RevolvingUtilizationOfUnsecuredLines",
    label: "Revolving credit utilization",
    hint: "ratio, ≥ 0",
    step: "0.01",
    min: 0,
  },
  { name: "age", label: "Age", hint: "years, 18–120", step: "1", min: 18, max: 120, int: true },
  {
    name: "NumberOfTime30-59DaysPastDueNotWorse",
    label: "Times 30–59 days past due",
    hint: "count",
    step: "1",
    min: 0,
    int: true,
  },
  { name: "DebtRatio", label: "Debt-to-income ratio", hint: "ratio, ≥ 0", step: "0.01", min: 0 },
  {
    name: "MonthlyIncome",
    label: "Monthly income",
    hint: "optional — leave blank if unverified",
    step: "1",
    min: 0,
    optional: true,
  },
  {
    name: "NumberOfOpenCreditLinesAndLoans",
    label: "Open credit lines & loans",
    hint: "count",
    step: "1",
    min: 0,
    int: true,
  },
  {
    name: "NumberOfTimes90DaysLate",
    label: "Times 90+ days late",
    hint: "count",
    step: "1",
    min: 0,
    int: true,
  },
  {
    name: "NumberRealEstateLoansOrLines",
    label: "Real-estate loans / lines",
    hint: "count",
    step: "1",
    min: 0,
    int: true,
  },
  {
    name: "NumberOfTime60-89DaysPastDueNotWorse",
    label: "Times 60–89 days past due",
    hint: "count",
    step: "1",
    min: 0,
    int: true,
  },
  {
    name: "NumberOfDependents",
    label: "Number of dependents",
    hint: "optional",
    step: "1",
    min: 0,
    optional: true,
    int: true,
  },
];

export const PRESETS = {
  "Low risk": {
    RevolvingUtilizationOfUnsecuredLines: 0.05,
    age: 52,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    DebtRatio: 0.2,
    MonthlyIncome: 12000,
    NumberOfOpenCreditLinesAndLoans: 6,
    NumberOfTimes90DaysLate: 0,
    NumberRealEstateLoansOrLines: 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    NumberOfDependents: 0,
  },
  Borderline: {
    RevolvingUtilizationOfUnsecuredLines: 0.55,
    age: 40,
    "NumberOfTime30-59DaysPastDueNotWorse": 1,
    DebtRatio: 0.42,
    MonthlyIncome: 4200,
    NumberOfOpenCreditLinesAndLoans: 9,
    NumberOfTimes90DaysLate: 0,
    NumberRealEstateLoansOrLines: 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    NumberOfDependents: 2,
  },
  "High risk": {
    RevolvingUtilizationOfUnsecuredLines: 0.99,
    age: 30,
    "NumberOfTime30-59DaysPastDueNotWorse": 4,
    DebtRatio: 0.9,
    MonthlyIncome: 1500,
    NumberOfOpenCreditLinesAndLoans: 3,
    NumberOfTimes90DaysLate: 3,
    NumberRealEstateLoansOrLines: 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 2,
    NumberOfDependents: 4,
  },
};

const emptyValues = () => Object.fromEntries(FIELDS.map((f) => [f.name, ""]));

function validateField(field, raw) {
  const value = raw.trim();
  if (value === "") {
    return field.optional ? null : "Required";
  }
  const n = Number(value);
  if (Number.isNaN(n)) return "Must be a number";
  if (field.int && !Number.isInteger(n)) return "Must be a whole number";
  if (field.min != null && n < field.min) return `Must be ≥ ${field.min}`;
  if (field.max != null && n > field.max) return `Must be ≤ ${field.max}`;
  return null;
}

export default function ApplicationForm({ onSubmit, submitting }) {
  const [values, setValues] = useState(emptyValues);
  const [errors, setErrors] = useState({});

  const setField = (name, v) => setValues((prev) => ({ ...prev, [name]: v }));

  const applyPreset = (name) => {
    const preset = PRESETS[name];
    setValues(Object.fromEntries(FIELDS.map((f) => [f.name, String(preset[f.name])])));
    setErrors({});
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const next = {};
    for (const field of FIELDS) {
      const err = validateField(field, values[field.name]);
      if (err) next[field.name] = err;
    }
    setErrors(next);
    if (Object.keys(next).length > 0) return;

    const payload = {};
    for (const field of FIELDS) {
      const raw = values[field.name].trim();
      payload[field.name] = raw === "" ? null : Number(raw);
    }
    onSubmit(payload);
  };

  return (
    <form onSubmit={handleSubmit} noValidate>
      <div className="presets">
        <span style={{ fontSize: 12.5, color: "var(--muted)", alignSelf: "center" }}>
          Load example:
        </span>
        {Object.keys(PRESETS).map((name) => (
          <button type="button" key={name} onClick={() => applyPreset(name)}>
            {name}
          </button>
        ))}
      </div>

      <div className="form-grid">
        {FIELDS.map((field) => (
          <div className="field" key={field.name}>
            <label htmlFor={field.name}>
              {field.label} <span className="hint">({field.hint})</span>
            </label>
            <input
              id={field.name}
              name={field.name}
              type="number"
              inputMode="decimal"
              step={field.step}
              min={field.min}
              max={field.max}
              value={values[field.name]}
              onChange={(e) => setField(field.name, e.target.value)}
              aria-invalid={errors[field.name] ? "true" : undefined}
            />
            {errors[field.name] && <div className="err">{errors[field.name]}</div>}
          </div>
        ))}
      </div>

      <div className="form-actions">
        <button type="submit" className="btn primary" disabled={submitting}>
          {submitting ? "Scoring…" : "Run review"}
        </button>
        <span style={{ fontSize: 12.5, color: "var(--muted)" }}>
          Runs data → model → SHAP → notice and saves an audit record.
        </span>
      </div>
    </form>
  );
}
