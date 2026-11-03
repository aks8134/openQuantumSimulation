"""Independently audit actual shot counts and six point estimates (stdlib only)."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def marginal(counts, mask):
    total = sum(counts.values())
    require(total >= 2 and all(isinstance(v, int) and v >= 0 for v in counts.values()), "invalid counts")
    require(all(len(k) == 10 and set(k) <= {"0", "1"} for k in counts), "invalid bitstrings")
    return [sum(((int(k, 2) ^ mask) >> j & 1)*v for k, v in counts.items())/total for j in range(10)]


def mean(groups):
    return [sum(v)/len(v) for v in zip(*groups)]


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    options = parser.parse_args(arguments)
    folder = options.folder
    report = json.loads((folder/"analysis.json").read_text())
    cfg = report["configuration"]
    root = Path(__file__).resolve().parents[2]
    require(hashlib.sha256(Path(__file__).with_name("fake_kingston_zne.py").read_bytes()).hexdigest() == cfg["source_code_sha256"], "runner changed")
    for filename, expected in cfg["implementation_sha256"].items():
        require(hashlib.sha256((root/filename).read_bytes()).hexdigest() == expected, "implementation changed")
    require(cfg == json.loads((folder/"configuration.json").read_text()), "analysis/config mismatch")
    source = json.loads((folder/"source.json").read_text())
    require(digest(source) == cfg["source_sha256"], "source hash mismatch")
    require(digest(json.loads((folder/"noise_model.json").read_text())) == cfg["noise_model_sha256"], "noise hash mismatch")
    control = {}
    values = {}
    shots = circuits = 0
    files = [folder/f"control_{name}.json" for name in ("all_zero", "all_one", "initial_mixture")]
    files += [folder/f"trajectory_{j}.json" for j in range(cfg["trajectory_count"])]
    for path in files:
        unit = json.loads(path.read_text())
        require(len(unit["records"]) == len(unit["observations"]), "missing observations")
        require(unit["source_sha256"] == cfg["source_sha256"], "unit source changed")
        require(unit["noise_model_sha256"] == cfg["noise_model_sha256"], "unit noise changed")
        require(hashlib.sha256((folder/unit["qpy_file"]).read_bytes()).hexdigest() == unit["qpy_sha256"], "QPY hash mismatch")
        require(all(group["aer_metadata"]["method"] == "statevector" for group in unit["simulation"]), "unexpected Aer method")
        for row, observation in zip(unit["records"], unit["observations"]):
            counts = observation["outcome_counts"]
            require(sum(counts.values()) == row["shots"], "actual shots differ from plan")
            shots += sum(counts.values())
            circuits += 1
            pop = marginal(counts, row["measurement_mask"])
            if row["kind"] == "control":
                control.setdefault((row["arm"], row["control"]), []).append(pop)
            else:
                key = (row["trajectory"], "raw" if row["arm"] == "raw" else row["factor_index"])
                values.setdefault(key, []).append(pop)
                require(row["metrics"]["operation_counts"]["cz"] == 880+2*row["folds"], "CZ count mismatch")
                require(row["metrics"]["operation_counts"]["reset"] == 44, "reservoir resets changed")
    require((circuits, shots) == (cfg["total_circuits"], cfg["total_shots"]), "workload mismatch")
    K = cfg["randomizations"]
    require(circuits == (1+3*K)*cfg["trajectory_count"]+3*(1+K), "incomplete circuit groups")
    for path in range(cfg["trajectory_count"]):
        for arm in ("raw", 0, 1, 2):
            require(len(values[path, arm]) == (1 if arm == "raw" else K), "missing randomizations")
    for arm in ("raw", "twirled"):
        for prep in ("all_zero", "all_one", "initial_mixture"):
            require(len(control[arm, prep]) == (1 if arm == "raw" else K), "missing controls")
    arms = {arm:mean([mean(values[path, arm]) for path in range(cfg["trajectory_count"])]) for arm in ("raw", 0, 1, 2)}
    weights = [17/6, 1/3, -13/6]
    require(all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(weights, report["statistics"]["linear_zne"]["weights"])), "ZNE weights changed")
    profiles = {"raw":arms["raw"], "twirled":arms[0],
                "zne":[sum(weights[k]*arms[k][j] for k in range(3)) for j in range(10)]}
    for name in list(profiles):
        arm = "raw" if name == "raw" else "twirled"
        a, b = [mean(control[arm, prep]) for prep in ("all_zero", "all_one")]
        require(min(y-x for x, y in zip(a, b)) >= .5, "insufficient calibration contrast")
        profiles[name+"_spam_corrected"] = [(p-x)/(y-x) for p, x, y in zip(profiles[name], a, b)]
    exact = [sum(c[j][j] for c in source["final_covariances_real_gauge"])/cfg["trajectory_count"] for j in range(10)]
    for name, populations in profiles.items():
        stored = report["statistics"]["profiles"][name]
        require(all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(populations, stored["populations"])), "population mismatch")
        error = [abs(a-b) for a, b in zip(populations, exact)]
        for key, actual in [("ideal_mae", sum(error)/10), ("ideal_maximum", max(error)),
                            ("physical_maximum", max(abs(a-.25) for a in populations))]:
            require(math.isclose(actual, stored[key], abs_tol=1e-12), "error metric mismatch")
    print(f"Verified {cfg['trajectory_count']} paths, {circuits} circuits, {shots:,} actual shots; "
          "source/noise/QPY hashes, reset/CZ counts, all six population profiles and their errors.")


if __name__ == "__main__":
    main()
