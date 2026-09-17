"""Sanity checks for giordano_snr.py, same spirit as test_tang_features.py --
targeted checks against signals with known expected behavior, run before
wiring into the CV pipeline."""

import numpy as np
from giordano_snr import compute_snr_db, _estimate_cycle_duration_samples

FS = 1000.0
rng = np.random.default_rng(42)


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    return cond


def make_heartbeat(fs, duration=10, hr_hz=1.2, noise_std=0.02):
    t = np.arange(0, duration, 1 / fs)
    hb = np.zeros(len(t))
    period = int(fs / hr_hz)
    for start in range(0, len(hb) - 20, period):
        hb[start:start + 10] += np.hanning(10) * 3
        hb[start + 30:start + 40] += np.hanning(10) * 2
    return hb + rng.normal(0, noise_std, len(t)), period


def main():
    all_ok = True

    clean, true_period = make_heartbeat(FS, noise_std=0.02)
    est_period = _estimate_cycle_duration_samples(clean, FS)
    all_ok &= check(
        "estimated cycle duration is within 20% of true period",
        abs(est_period - true_period) / true_period < 0.20,
        f"true={true_period} est={est_period}",
    )

    snr_clean = compute_snr_db(clean, FS)
    all_ok &= check(
        "clean heartbeat has high SNR",
        snr_clean > 10,
        f"got {snr_clean:.2f} dB",
    )

    pure_noise = rng.normal(0, 1, len(clean))
    snr_noise = compute_snr_db(pure_noise, FS)
    all_ok &= check(
        "pure noise has much lower SNR than clean heartbeat",
        snr_noise < snr_clean - 10,
        f"noise={snr_noise:.2f} dB vs clean={snr_clean:.2f} dB",
    )

    # Monotonicity: as we mix in more noise (RMS-scaled, matching the actual
    # pipeline's mix_rms), SNR should trend downward.
    heart, _ = make_heartbeat(FS, noise_std=0.02)
    noise_source = rng.normal(0, 1, len(heart))
    rms_h = np.sqrt(np.mean(heart ** 2))
    rms_n = np.sqrt(np.mean(noise_source ** 2))
    scale = rms_h / rms_n

    snrs_by_lambda = []
    for lam in [0, 0.5, 1, 5, 10]:
        mixed = heart + lam * (noise_source * scale)
        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed = mixed / peak
        snrs_by_lambda.append(compute_snr_db(mixed, FS))

    all_ok &= check(
        "SNR trends downward as mixing lambda increases",
        snrs_by_lambda[0] > snrs_by_lambda[-1],
        f"snrs={[round(s,2) for s in snrs_by_lambda]}",
    )

    # Edge cases: must not crash
    try:
        s = compute_snr_db(np.zeros(1000), FS)
        all_ok &= check("all-zero signal doesn't crash", np.isfinite(s) or s == -np.inf, f"got {s}")
    except Exception as e:
        all_ok &= check("all-zero signal doesn't crash", False, str(e))

    try:
        s = compute_snr_db(rng.normal(0, 1, 50), FS)  # very short signal
        all_ok &= check("very short signal doesn't crash", True, f"got {s}")
    except Exception as e:
        all_ok &= check("very short signal doesn't crash", False, str(e))

    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED -- do not proceed until fixed")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
