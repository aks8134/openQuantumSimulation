"""Pure saved-count diagnostics for the three population controls."""
from __future__ import annotations

from statistics import NormalDist
import math


def wilson_upper(probability, shots, z):
    denominator = 1+z*z/shots
    return (probability+z*z/(2*shots)+z*math.sqrt(probability*(1-probability)/shots+z*z/(4*shots*shots)))/denominator


def assess_controls(controls, max_error=.03, alpha=.05):
    """Conservative screening, not certification of evolved-state accuracy.

    Allocate alpha over 2n one-sided baseline bounds and 2n mixture tails.
    Mixture z tests use empirical binomial SEs and a normal approximation;
    the family adjustment is therefore nominal rather than an exact guarantee.
    """
    if not 0 < max_error < 1 or not 0 < alpha < 1:
        raise ValueError("invalid control screening thresholds")
    rows = {row["control"]: row for row in controls}
    if len(rows) != len(controls) or set(rows) != {"all_zero", "all_one", "initial_mixture"}:
        raise ValueError("exactly the three population controls are required")
    n = len(rows["all_zero"]["populations"])
    if n < 1 or any(len(row["populations"]) != n or row["shots"] < 2 for row in rows.values()):
        raise ValueError("inconsistent control dimensions or shot counts")
    mapping = rows["all_zero"]["physical_sites"]
    if any(row["physical_sites"] != mapping for row in rows.values()):
        raise ValueError("controls must measure the same physical sites")
    z = NormalDist().inv_cdf(1-alpha/(4*n))
    results, failed = [], []
    zero, one, mixture = (rows[name] for name in ("all_zero", "all_one", "initial_mixture"))
    for j in range(n):
        a, b, c = (row["populations"][j] for row in (zero, one, mixture))
        p = mixture["expected_populations"][j]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (a, b, c, p)):
            raise ValueError("control populations must be probabilities")
        if zero["expected_populations"][j] != 0 or one["expected_populations"][j] != 1:
            raise ValueError("zero/one controls have incorrect targets")
        upper_zero = wilson_upper(a, zero["shots"], z)
        upper_one = wilson_upper(1-b, one["shots"], z)
        prediction = (1-p)*a+p*b
        variance = (c*(1-c)/(mixture["shots"]-1)
                    +(1-p)**2*a*(1-a)/(zero["shots"]-1)
                    +p*p*b*(1-b)/(one["shots"]-1))
        residual = c-prediction
        se = math.sqrt(max(0., variance))
        reasons = []
        if upper_zero > max_error:
            reasons.append("zero-control error upper bound exceeds threshold")
        if upper_one > max_error:
            reasons.append("one-control error upper bound exceeds threshold")
        if abs(residual) > z*se+1e-12:
            reasons.append("initial mixture inconsistent with zero/one response")
        row = {"site_1based": j+1, "physical_qubit": mapping[j], "zero_population": a,
               "one_population": b, "mixture_population": c, "predicted_mixture": prediction,
               "mixture_residual": residual, "mixture_residual_se": se,
               "zero_error_upper_bound": upper_zero, "one_error_upper_bound": upper_one,
               "passed": not reasons, "reasons": reasons}
        results.append(row)
        if reasons:
            failed.append(j+1)
    return {"passed": not failed, "failed_sites_1based": failed, "sites": results,
            "max_baseline_error": max_error, "family_alpha": alpha, "nominal_z_cutoff": z,
            "interpretation": "Diagnostic screening of preparation/readout, with nominal family-adjusted mixture tests. Passing does not certify 0.02 accuracy or reservoir-reset performance."}


def markdown(assessment):
    lines = ["# Population controls", "", f"Diagnostic passed: **{assessment['passed']}**.",
        f"Maximum allowed baseline error upper bound: {assessment['max_baseline_error']:.3%}; nominal family alpha {assessment['family_alpha']}.", "",
        "| Site | Physical qubit | Zero | One | Mixture | Predicted mixture | Pass |",
        "|---:|---:|---:|---:|---:|---:|---|"]
    for row in assessment["sites"]:
        lines.append(f"| {row['site_1based']} | {row['physical_qubit']} | {row['zero_population']:.6f} | {row['one_population']:.6f} | {row['mixture_population']:.6f} | {row['predicted_mixture']:.6f} | {row['passed']} |")
    lines += ["", assessment["interpretation"], ""]
    for row in assessment["sites"]:
        if row["reasons"]:
            lines.append(f"Site {row['site_1based']} / qubit {row['physical_qubit']}: {'; '.join(row['reasons'])}.")
    return "\n".join(lines)+"\n"
