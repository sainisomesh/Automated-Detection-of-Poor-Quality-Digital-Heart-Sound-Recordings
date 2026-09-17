"""
Sanity checks for tang_features.py, run BEFORE wiring into the real CV
pipeline. These aren't a full test suite -- they're targeted checks that each
ported function behaves the way the published algorithm should on signals
whose expected behavior we know analytically, to catch porting bugs early
(wrong axis, wrong normalization, off-by-one in lag windows, etc).

Run: python3 test_tang_features.py
"""

import numpy as np
from tang_features import (
    remove_spike, pre_processing, get_envelope_from_stft, get_kurtosis,
    get_energy_ratio, get_max_axcor_coef, get_sampen_fast,
    get_degree_cycle, extract_features, FEATURE_NAMES,
)

FS = 1000.0
rng = np.random.default_rng(42)


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    return cond


def main():
    all_ok = True

    # ── remove_spike ──
    x = rng.normal(0, 1, 5000)
    x[100] = 1000.0  # inject an obvious spike
    y = remove_spike(x)
    all_ok &= check(
        "remove_spike clips injected outlier",
        abs(y[100]) < 100.0,
        f"spike went from {x[100]:.1f} to {y[100]:.2f}",
    )
    all_ok &= check(
        "remove_spike leaves normal samples ~unchanged",
        np.allclose(y[:100], x[:100]),
    )

    # ── get_kurtosis ──
    gauss = rng.normal(0, 1, 200_000)
    k_gauss = get_kurtosis(gauss)
    all_ok &= check(
        "kurtosis of Gaussian noise ~3",
        2.8 < k_gauss < 3.2,
        f"got {k_gauss:.3f}",
    )

    impulsive = rng.normal(0, 1, 200_000)
    impulsive[::500] += rng.normal(0, 20, len(impulsive[::500]))
    k_imp = get_kurtosis(impulsive)
    all_ok &= check(
        "kurtosis of impulsive signal >> Gaussian",
        k_imp > k_gauss * 2,
        f"got {k_imp:.3f} vs gaussian {k_gauss:.3f}",
    )

    # ── get_energy_ratio ──
    t = np.arange(0, 10, 1 / FS)
    low_tone = np.sin(2 * np.pi * 50 * t)     # inside [24,144] band
    high_tone = np.sin(2 * np.pi * 400 * t)   # inside [200, fs/2] band
    r_low_in_low = get_energy_ratio(low_tone, (24, 144), FS)
    r_low_in_high = get_energy_ratio(low_tone, (200, FS / 2), FS)
    all_ok &= check(
        "energy ratio: 50Hz tone mostly in low band, not high band",
        r_low_in_low > 0.8 and r_low_in_high < 0.05,
        f"low-band={r_low_in_low:.3f} high-band={r_low_in_high:.3f}",
    )
    r_high_in_high = get_energy_ratio(high_tone, (200, FS / 2), FS)
    all_ok &= check(
        "energy ratio: 400Hz tone mostly in high band",
        r_high_in_high > 0.8,
        f"got {r_high_in_high:.3f}",
    )

    # ── get_max_axcor_coef ──
    period_samples = 700  # 0.7s period -> within [0.3fs, 2fs] lag search window
    periodic = np.sin(2 * np.pi * t / (period_samples / FS))
    full = np.correlate(periodic, periodic, mode="full")
    single = full[len(periodic) - 1:] / full[len(periodic) - 1]
    max_coef = get_max_axcor_coef(single, FS)
    all_ok &= check(
        "max autocorr coef of periodic signal is strongly positive",
        max_coef > 0.7,
        f"got {max_coef:.3f}",
    )

    # ── get_sampen_fast ──
    sine = np.sin(2 * np.pi * 2 * t)
    noise = rng.normal(0, 1, len(t))
    se_sine = get_sampen_fast(sine, 2, 0.2)
    se_noise = get_sampen_fast(noise, 2, 0.2)
    all_ok &= check(
        "sample entropy: regular sine << random noise",
        se_sine < se_noise,
        f"sine={se_sine:.3f} noise={se_noise:.3f}",
    )

    # ── get_degree_cycle ──
    heart_rate_hz = 1.2  # ~72 bpm envelope modulation
    periodic_envelope = 1 + 0.8 * np.sin(2 * np.pi * heart_rate_hz * t)
    d_periodic = get_degree_cycle(periodic_envelope, 0.3, 2.5, FS)
    d_noise = get_degree_cycle(rng.normal(0, 1, len(t)), 0.3, 2.5, FS)
    all_ok &= check(
        "degree of periodicity: periodic signal >> noise",
        d_periodic > d_noise * 1.5,
        f"periodic={d_periodic:.3f} noise={d_noise:.3f}",
    )

    # ── full pipeline: pre_processing + extract_features on synthetic PCG-like signal ──
    heartbeat = np.zeros(len(t))
    beat_period = int(FS / heart_rate_hz)
    for start in range(0, len(heartbeat) - 20, beat_period):
        heartbeat[start:start + 10] += np.hanning(10) * 3   # S1
        heartbeat[start + 30:start + 40] += np.hanning(10) * 2  # S2
    synth_pcg = heartbeat + rng.normal(0, 0.05, len(t))

    processed = pre_processing(synth_pcg, FS)
    all_ok &= check(
        "pre_processing output is finite and non-constant",
        np.all(np.isfinite(processed)) and np.std(processed) > 0,
    )

    feats = extract_features(processed, FS)
    all_ok &= check(
        "extract_features returns 10 finite features",
        feats.shape == (10,) and np.all(np.isfinite(feats)),
        f"features={dict(zip(FEATURE_NAMES, np.round(feats, 4)))}",
    )

    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED -- do not proceed until fixed")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
