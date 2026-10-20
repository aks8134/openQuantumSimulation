"""Three-basis reconstruction and uncertainty accounting, independent of experiments."""
import numpy as np
import qem

FAMILIES = ("populations", "xy_correlations", "excitation_flows")


def qem_value(result):
    if isinstance(result, qem.Err):
        raise RuntimeError(f"{result.error.technique}: {result.error.message}")
    return result.value


def parity(record, bits):
    counts = record["counts"]
    serialized = record["diagnostic"].get("readout_model")
    if serialized is not None:
        model = qem.ReadoutModel(tuple(qem.ReadoutBlock(tuple(block["bits"]),
            tuple(tuple(row) for row in block["assignment"])) for block in serialized["blocks"]),
            source=serialized["source"])
        corrected = qem_value(qem.mitigate_parity(counts, model, tuple(bits)))
        return corrected.estimate, (corrected.standard_error or 0.0)**2
    shots = sum(counts.values())
    def outcome(key):
        value = "".join(key.split())
        return int(value, 16 if value.lower().startswith("0x") else 2)
    mean = sum((-1 if sum((outcome(key) >> bit) & 1 for bit in bits) % 2 else 1) * count
               for key, count in counts.items()) / shots
    return mean, max(0.0, 1 - mean**2) / shots


def _summary(values, variances):
    """Last two axes are trajectory and fold/twirl realization."""
    R, F = values.shape[-2:]
    mean = values.mean(axis=(-2, -1))
    shot = variances.sum(axis=(-2, -1)) / (R * F)**2
    fold_means = values.mean(axis=-1)
    fold_shot = variances.sum(axis=-1) / F**2
    within_fold = np.zeros_like(fold_means) if F == 1 else np.maximum(
        values.var(axis=-1, ddof=1) - variances.mean(axis=-1), 0) / F
    folding = within_fold.sum(axis=-1) / R**2
    trajectory = np.zeros_like(mean) if R == 1 else np.maximum(
        fold_means.var(axis=-1, ddof=1) - (fold_shot + within_fold).mean(axis=-1), 0) / R
    return mean, shot, folding, trajectory


def reconstruct(records, model, config):
    times = sorted({record["time"] for record in records})
    R, F, S = config["trajectories"], config["fold_repetitions"], len(config["zne_scale_factors"])
    lookup = {(row["time"], row["trajectory_index"], row["basis"], row["scale_index"],
               row["fold_repetition"]): row for row in records}
    if len(lookup) != len(records) or len(records) != len(times) * R * F * S * 3:
        raise ValueError("duplicate or incomplete time/trajectory/basis/scale/fold groups")
    shapes = ((model.n_qubits, len(times), S, R, F),
              (model.n_qubits - 1, len(times), S, R, F))
    values = [np.zeros(shapes[0]), np.zeros(shapes[1]), np.zeros(shapes[1])]
    variances = [np.zeros_like(value) for value in values]
    for ti, t in enumerate(times):
        for si in range(S):
            for r in range(R):
                for f in range(F):
                    z, x, mixed = (lookup[(t, r, basis, si, f)] for basis in ("Z", "X", "XY"))
                    for site in range(model.n_qubits):
                        mean, variance = parity(z, (site,))
                        values[0][site, ti, si, r, f] = 0.5 * (1 - mean)
                        variances[0][site, ti, si, r, f] = 0.25 * variance
                    for bond in range(model.n_qubits - 1):
                        mean, variance = parity(x, (bond, bond + 1))
                        values[1][bond, ti, si, r, f] = 2 * mean
                        variances[1][bond, ti, si, r, f] = 4 * variance
                        mean, variance = parity(mixed, (bond, bond + 1))
                        coefficient = -model.J if bond % 2 == 0 else model.J
                        values[2][bond, ti, si, r, f] = coefficient * mean
                        variances[2][bond, ti, si, r, f] = coefficient**2 * variance
    plan = qem.ZNEPlan(tuple(config["zne_scale_factors"]),
                       "none" if config["no_zne"] else config["zne_inference"],
                       degree=config["zne_polynomial_order"])
    # Fitting logic is owned by qem; fixed-fit weights allow vectorized application.
    weights = np.asarray(qem_value(qem.extrapolate(plan, [0.0] * S)).weights)
    summaries = [_summary(v, variance) for v, variance in zip(values, variances)]
    means = [summary[0] for summary in summaries]
    measured = [np.sum(mean * weights, axis=-1) for mean in means]
    errors = {name: [np.sqrt(np.sum(summary[index] * weights**2, axis=-1)) for summary in summaries]
              for name, index in (("shot", 1), ("folding", 2), ("trajectory", 3))}
    errors["total"] = [np.sqrt(sum(errors[name][family]**2 for name in ("shot", "folding", "trajectory")))
                       for family in range(3)]
    return {"times": times, "observables": family_payload(measured),
            "standard_errors": {name: family_payload(arrays) for name, arrays in errors.items()},
            "scale_observables": family_payload(means),
            "scale_standard_errors": {name: family_payload([np.sqrt(summary[index]) for summary in summaries])
                for name, index in (("shot", 1), ("folding", 2), ("trajectory", 3))},
            "extrapolation_weights": weights.tolist()}


def family_payload(arrays):
    return {name: array.tolist() for name, array in zip(FAMILIES, arrays)}


def resource_summary(records, scales):
    result = []
    for index, scale in enumerate(scales):
        rows = [row for row in records if row["scale_index"] == index]
        if not rows:
            continue
        result.append({"requested_scale_factor": scale,
            "mean_actual_physical_scale_factor": float(np.mean([r["diagnostic"]["actual_physical_scale_factor"] for r in rows])),
            "mean_original_explicit_operation_count": float(np.mean([r["original_operation_count"] for r in rows])),
            "mean_original_explicit_depth": float(np.mean([r["original_depth"] for r in rows])),
            "mean_base_transpiled_operation_count": float(np.mean([r["diagnostic"]["base"]["operations"] for r in rows])),
            "mean_base_transpiled_depth": float(np.mean([r["diagnostic"]["base"]["depth"] for r in rows])),
            "mean_submitted_operation_count": float(np.mean([r["compiled_operation_count"] for r in rows])),
            "mean_submitted_depth": float(np.mean([r["compiled_depth"] for r in rows]))})
    return result
