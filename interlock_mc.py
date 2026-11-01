#!/usr/bin/env python3
"""interlock_mc -- compact barrier-integrity Monte Carlo for a laser-scanned
aircraft-fuelling permissive interlock (MyUncle-style single file; Leon Sandler).

Three mutually cross-checking levels, as in MyUncle:
 (A) closed-form per-event misfuelling hazard for each barrier configuration
     (stress-dependent human error, beta-factor common cause, relay PFD);
 (B) mean-field reduction of interlock *compromise* (bypass / habitual override):
         dI/dtau = mu(I)(1-I) - rho*I ,  rho = R / lam_eff ,  mu(I)=1+beta I^h/(K^h+I^h)
     identical in form to MyUncle's maintenance equation; I = compromised fraction;
 (C) fleet Monte Carlo (trucks x days) with Rao-Blackwellised event hazards, outer-loop
     parameter uncertainty (lognormal priors), PRCC sensitivity, and a crude Bernoulli
     event simulator used to validate (A).

All parameters are ASSUMED priors for an ideation-stage (TRL 1-3) concept, not field data.
Needs numpy, scipy, matplotlib.   CLI:  python interlock_mc.py --demo | --reproduce
"""
from __future__ import annotations
import argparse, json, math, os
from dataclasses import dataclass, asdict, field
import numpy as np
from scipy.stats import rankdata

# ----------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------
@dataclass
class Params:
    # exposure
    n_trucks: int = 5
    events_per_truck_day: float = 20.0
    days: int = 365
    # human error generation (per fuelling event) and stress feedback
    lam0: float = 1.0e-4          # product-mismatch *attempt* probability, unstressed
    beta_s: float = 4.0; h_s: float = 3.0; K_s: float = 0.6     # Hill stress amplification
    m_adm0: float = 0.01          # administrative-check miss probability, unstressed
    stress_gain: float = 4.0      # m_adm = m_adm0 (1 + stress_gain S)
    m_ovr0: float = 0.02          # supervisor-override verification miss, unstressed
    # depot / loading error that a records-based interlock cannot see
    p_load: float = 2.0e-6; m_load_adm: float = 0.3
    # interlock chain
    p_u0: float = 0.003; p_u1: float = 0.030    # tag-unreadable prob = p_u0 + p_u1 W
    p_assoc: float = 5.0e-4       # wrong-tag / stale-record / manifest association error
    p_pass_assoc: float = 0.6     # association error still yields a compatible-looking permit
    lam_DU: float = 2.0e-7        # relay dangerous-undetected failure rate [1/h]
    T_proof: float = 8760.0       # proof-test interval [h]
    p_sw: float = 1.0e-5          # controller software dangerous fault per demand
    beta_cc: float = 0.05         # common-cause beta between admin check and interlock
    p_false: float = 0.005        # false-mismatch nuisance (late aircraft swap etc.)
    # strategy switches
    interlock: bool = True
    n_tags: int = 1               # 2 -> redundant tags
    beta_tag: float = 0.10        # common-cause for dual tags (same fouling event)
    d_sens: float = 0.0           # inline product-verification detection probability
    redundant_relay: bool = False # 1oo2 relay (PFD ~ beta_r * single PFD)
    beta_r: float = 0.1
    # compromise (bypass) dynamics
    nu: float = 2.0e-3            # defeat probability per nuisance event
    beta_c: float = 20.0; h_c: float = 6.0; K_c: float = 0.5   # peer-normalisation Hill (MyUncle defaults)
    T_detect: float = 90.0        # days to detect + restore a compromised truck
    # operations / delay
    f_rescan: float = 0.7; t_rescan: float = 20.0; t_ovr_med: float = 240.0; sig_ovr: float = 0.6
    # environment
    S_a: float = 2.0; S_b: float = 5.0            # daily stress ~ Beta(a,b)
    W_amp: float = 0.35; W_mean: float = 0.35; W_phase: float = 0.0
    W_mult: float = 1.0                           # multiplies tag-unreadable prob (shock studies)
    # v2 additions
    complacency: float = 1.0      # multiplier on administrative miss when an interlock is present (risk compensation)
    p_sens_false: float = 0.002   # product-sensor false-veto probability per fuelling (only if d_sens > 0)
    label: str = "default"

    def copy(self, **kw):
        d = asdict(self); d.update(kw); return Params(**d)

PRESETS = {
    "S0_baseline_admin": Params(label="S0 baseline (admin checks)", interlock=False),
    "S1_interlock_basic": Params(label="S1 interlock as proposed", T_detect=90.0),
    "S2_interlock_audit": Params(label="S2 + automated tamper analytics", T_detect=7.0),
    "S3_interlock_full": Params(label="S3 + dual tags, 1oo2 relay, product sensor",
                                T_detect=7.0, n_tags=2, redundant_relay=True, d_sens=0.95),
}

# ----------------------------------------------------------------------------------
# (A) closed-form event hazards (vectorised over S, W, comp)
# ----------------------------------------------------------------------------------
def hill(x, beta, h, K):
    x = np.asarray(x, float); return beta * x**h / (K**h + x**h)

def lam_attempt(P, S):         return P.lam0 * (1.0 + hill(S, P.beta_s, P.h_s, P.K_s))
def m_admin(P, S, with_interlock=False):
    k = P.complacency if with_interlock else 1.0
    return np.clip(P.m_adm0 * k * (1.0 + P.stress_gain * S), 0, 1)
def m_override(P, S):          return np.clip(P.m_ovr0 * (1.0 + P.stress_gain * S), 0, 1)

def p_unread(P, W):
    pu = np.clip((P.p_u0 + P.p_u1 * np.asarray(W, float)) * P.W_mult, 0, 1)
    if P.n_tags == 2: pu = pu * (P.beta_tag + (1 - P.beta_tag) * pu)
    return pu

def relay_pfd(P):
    pfd = P.lam_DU * P.T_proof / 2.0
    return pfd * P.beta_r if P.redundant_relay else pfd

def p_nuisance(P, W):
    sf = P.p_sens_false if np.any(np.asarray(P.d_sens) > 0) else 0.0
    return np.clip(p_unread(P, W) + P.p_false + sf, 0, 1)

def p_interlock_pass(P, S, W, comp):
    """P(interlock permits | a true product mismatch is attempted)."""
    pu = p_unread(P, W)
    via_override = pu * m_override(P, S)
    via_assoc = (1 - pu) * P.p_assoc * P.p_pass_assoc
    via_fail = relay_pfd(P) + P.p_sw
    p = np.minimum(1.0, via_override + via_assoc + via_fail)
    return np.where(comp, 1.0, p)

def event_hazard(P, S, W, comp):
    """Per-event probability of an uncorrected misfuelling."""
    S = np.asarray(S, float); W = np.asarray(W, float); comp = np.asarray(comp, bool)
    lam = lam_attempt(P, S); ma = m_admin(P, S, P.interlock)
    if P.interlock:
        pi = p_interlock_pass(P, S, W, comp)
        joint = P.beta_cc * np.minimum(pi, ma) + (1 - P.beta_cc) * pi * ma
        sens = P.d_sens * (~comp)
    else:
        joint = ma; sens = 0.0 * ma
    return lam * joint + P.p_load * P.m_load_adm * (1 - sens)

def nominal_hazard(P, S=None, W=None):
    """Closed-form hazard at mean environment, no compromise."""
    S = P.S_a / (P.S_a + P.S_b) if S is None else S
    W = P.W_mean if W is None else W
    return float(event_hazard(P, S, W, False))

def hazard_breakdown(P, S=None, W=None):
    """Nominal per-event hazard split by path (per 1e6 events)."""
    S = P.S_a / (P.S_a + P.S_b) if S is None else S; W = P.W_mean if W is None else W
    lam = float(lam_attempt(P, S)); ma = float(m_admin(P, S, P.interlock)); pu = float(p_unread(P, W))
    if not P.interlock:
        return dict(attempt_path=1e6 * lam * ma, load_path=1e6 * P.p_load * P.m_load_adm)
    ov = pu * float(m_override(P, S)); asc = (1 - pu) * P.p_assoc * P.p_pass_assoc; fl = relay_pfd(P) + P.p_sw
    pi = min(1.0, ov + asc + fl); joint = P.beta_cc * min(pi, ma) + (1 - P.beta_cc) * pi * ma
    tot = ov + asc + fl
    return dict(override_path=1e6 * lam * joint * ov / tot, association_path=1e6 * lam * joint * asc / tot,
                hardware_path=1e6 * lam * joint * fl / tot,
                load_path=1e6 * P.p_load * P.m_load_adm * (1 - P.d_sens))

def delay_stats(P, W):
    pn = p_nuisance(P, W)
    mean_ovr = P.t_ovr_med * math.exp(P.sig_ovr**2 / 2)
    mean_delay = pn * (P.f_rescan * P.t_rescan + (1 - P.f_rescan) * mean_ovr)      # s / event
    # P(delay > 300 s) per event: only override branch can exceed
    from scipy.stats import norm
    tail = 1 - norm.cdf((math.log(300) - math.log(P.t_ovr_med)) / P.sig_ovr)
    return mean_delay, pn * (1 - P.f_rescan) * tail

# ----------------------------------------------------------------------------------
# (B) mean-field compromise model
# ----------------------------------------------------------------------------------
class MeanField:
    def __init__(self, P): self.P = P
    def mu(self, I):
        P = self.P; I = np.asarray(I, float); return 1 + hill(I, P.beta_c, P.h_c, P.K_c)
    def rhs(self, I, rho): return self.mu(I) * (1 - I) - rho * I
    def fixed_points(self, rho, n=6000):
        I = np.linspace(1e-4, 1 - 1e-4, n); g = self.rhs(I, rho)
        return I[np.where(np.diff(np.sign(g)) != 0)[0]]
    def stable(self, I, rho, d=1e-5): return (self.rhs(I + d, rho) - self.rhs(I - d, rho)) / (2 * d) < 0
    def bistable_window(self, n=8000):
        I = np.linspace(1e-3, 0.999, n); F = self.mu(I) * (1 - I) / I
        t = np.where(np.diff(np.sign(np.diff(F))) != 0)[0]
        if len(t) >= 2: r = sorted(F[t]); return float(r[0]), float(r[-1])
        return None
    def lam_eff(self, W=None, events=None):
        P = self.P; W = P.W_mean if W is None else W
        events = P.events_per_truck_day if events is None else events
        return P.nu * events * float(p_nuisance(P, W))        # isolated-truck defeat hazard [1/day]
    def rho(self, W=None): return (1.0 / self.P.T_detect) / max(self.lam_eff(W), 1e-12)
    def branches(self, rho):
        fp = self.fixed_points(rho); return [float(x) for x in fp if self.stable(x, rho)]

# ----------------------------------------------------------------------------------
# (C) fleet Monte Carlo
# ----------------------------------------------------------------------------------
PRIORS = {   # name: (median, geometric-SD factor); lognormal, clipped to valid ranges
    "lam0": (1e-4, 3.0), "m_adm0": (0.01, 2.5), "m_ovr0": (0.02, 2.5), "p_assoc": (5e-4, 3.0),
    "p_u0": (0.003, 2.0), "p_u1": (0.03, 2.0), "lam_DU": (2e-7, 3.0), "p_sw": (1e-5, 3.0),
    "beta_cc": (0.05, 2.0), "nu": (2e-3, 3.0), "p_load": (2e-6, 3.0), "T_detect": (None, 1.5),
    "p_false": (0.005, 2.0),
}
def draw_params(P, z):
    """Map standard-normal draws z (len(PRIORS)) to a parameter set; the same z reused across
    configurations gives paired (common-random-number) comparisons."""
    d = {}
    for i, (k, (med, gsd)) in enumerate(PRIORS.items()):
        v = getattr(P, k) * math.exp(z[i] * math.log(gsd))
        d[k] = float(min(v, 0.5)) if k in ("beta_cc", "m_adm0", "m_ovr0", "p_assoc", "p_u0", "p_u1", "p_false", "p_load") else float(v)
    return P.copy(**d)

def simulate_fleet(P, rng, shock=None):
    """One stochastic year for one fleet. shock=(day0, day1, W_mult) scales tag-unreadable prob.
    Returns dict of annual metrics and (optionally) the daily compromised-fraction series."""
    N, D = P.n_trucks, P.days
    comp = np.zeros(N, bool); mf = MeanField(P)
    exp_mis = 0.0; n_events = 0.0; n_nuis = 0.0; delay_s = 0.0; dly5 = 0.0
    series = np.zeros(D); S_all = rng.beta(P.S_a, P.S_b, D)
    t = np.arange(D)
    W_all = np.clip(P.W_mean + P.W_amp * np.sin(2 * np.pi * t / 365 + P.W_phase)
                    + rng.normal(0, 0.1, D), 0, 1)
    p_rep = 1.0 / P.T_detect
    for d in range(D):
        Pd = P if not (shock and shock[0] <= d < shock[1]) else P.copy(W_mult=P.W_mult * shock[2])
        S, W = S_all[d], W_all[d]
        ev = rng.poisson(P.events_per_truck_day, N)
        hz = event_hazard(Pd, S, W, comp)
        exp_mis += float(np.sum(ev * hz)); n_events += float(ev.sum())
        if P.interlock:
            pn = float(p_nuisance(Pd, W)); nn = rng.binomial(ev, pn)
            n_nuis += float(nn.sum()); md, d5 = delay_stats(Pd, W)
            delay_s += md * float(ev.sum()); dly5 += d5 * float(ev.sum())
            I = comp.mean()
            mu = 1.0 + float(hill(I, P.beta_c, P.h_c, P.K_c))
            h_def = 1 - np.exp(-P.nu * nn * mu)              # nu per nuisance event, peer-amplified
            new = (~comp) & (rng.random(N) < h_def)
            rep = comp & (rng.random(N) < p_rep)
            comp = (comp | new) & ~rep
        series[d] = comp.mean()
    return dict(pmfe=1e6 * exp_mis / n_events, exp_mis=exp_mis, events=n_events,
                nuis_rate=n_nuis / n_events, comp_mean=float(series.mean()),
                comp_end=float(series[-1]), collapse=float(series.mean() > 0.5),
                delay_s=delay_s / n_events, p_delay5=dly5 / n_events, series=series)

def _prior_z(Z, prior):
    from scipy.stats import norm
    if prior == "loguniform":      # same SD as the log-normal, bounded support
        return (2 * norm.cdf(Z) - 1) * math.sqrt(3.0)
    return Z

def draw_params_batch(P, Z, prior="lognormal", gsd_scale=1.0):
    """Z: (n, len(PRIORS)) standard normals -> Params whose uncertain fields are (n,1) arrays."""
    Zp = _prior_z(Z, prior); d = {}
    for i, (k, (med, gsd)) in enumerate(PRIORS.items()):
        v = getattr(P, k) * np.exp(Zp[:, i] * math.log(gsd) * gsd_scale)
        if k in ("beta_cc", "m_adm0", "m_ovr0", "p_assoc", "p_u0", "p_u1", "p_false", "p_load"): v = np.minimum(v, 0.5)
        d[k] = v.reshape(-1, 1)
    return P.copy(**d)

def simulate_batch(P, n, rng, shock=None):
    """Vectorised fleet-years: n independent replicates x P.n_trucks vehicles x P.days days.
    Uncertain parameters in P may be (n,1) arrays (one parameter set per replicate)."""
    N, D = int(P.n_trucks), int(P.days)
    comp = np.zeros((n, N), bool)
    S_all = rng.beta(P.S_a, P.S_b, (D, n, 1)); t = np.arange(D)[:, None, None]
    W_all = np.clip(P.W_mean + P.W_amp * np.sin(2 * np.pi * t / 365 + P.W_phase) + rng.normal(0, 0.1, (D, n, 1)), 0, 1)
    exp_mis = np.zeros(n); events = np.zeros(n); nuis = np.zeros(n); delay = np.zeros(n); d5 = np.zeros(n)
    series = np.zeros((D, n)); p_rep = 1.0 / P.T_detect; wm0 = P.W_mult
    for d in range(D):
        P.W_mult = wm0 * (shock[2] if (shock and shock[0] <= d < shock[1]) else 1.0)
        S, W = S_all[d], W_all[d]
        ev = rng.poisson(P.events_per_truck_day, (n, N)); evs = ev.sum(1)
        exp_mis += (ev * event_hazard(P, S, W, comp)).sum(1); events += evs
        if P.interlock:
            pn = p_nuisance(P, W); nn = rng.binomial(ev, pn); nuis += nn.sum(1)
            md, dd = delay_stats(P, W); delay += (md * evs[:, None]).ravel() if np.ndim(md) else md * evs
            d5 += (dd * evs[:, None]).ravel() if np.ndim(dd) else dd * evs
            I = comp.mean(1, keepdims=True); mu = 1.0 + hill(I, P.beta_c, P.h_c, P.K_c)
            new = (~comp) & (rng.random((n, N)) < 1 - np.exp(-P.nu * nn * mu))
            rep = comp & (rng.random((n, N)) < p_rep)
            comp = (comp | new) & ~rep
        series[d] = comp.mean(1)
    P.W_mult = wm0
    cm = series.mean(0)
    return dict(pmfe=1e6 * exp_mis / events, exp_mis=exp_mis, events=events, nuis_rate=nuis / events,
                comp_mean=cm, comp_end=series[-1], collapse=(cm > 0.5).astype(float),
                delay_s=delay / events, p_delay5=d5 / events, series=series.T)

def run_ensemble(P, n_rep=300, seed=0, shock=None, keep_series=False, n_stoch=1, prior="lognormal", gsd_scale=1.0):
    """n_rep replicates = (n_rep // n_stoch) parameter sets x n_stoch stochastic fleet-years each.
    Parameter draws depend only on `seed`, so they are common across configurations (paired)."""
    n_par = max(n_rep // n_stoch, 1); n = n_par * n_stoch
    Z = np.random.default_rng(seed).normal(size=(n_par, len(PRIORS))); Zr = np.repeat(Z, n_stoch, axis=0)
    Pb = draw_params_batch(P, Zr, prior, gsd_scale); rng = np.random.default_rng(seed + 10_000)
    out = simulate_batch(Pb, n, rng, shock)
    out["params"] = np.column_stack([np.asarray(getattr(Pb, k)).reshape(-1) for k in PRIORS])
    out["n_par"] = n_par; out["n_stoch"] = n_stoch
    if not keep_series: out.pop("series")
    return out

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def variance_decomposition(pmfe, n_par, n_stoch):
    """Share of Var[log10 PMFE] attributable to parameter uncertainty (between parameter sets)."""
    y = np.log10(np.maximum(pmfe, 1e-12)).reshape(n_par, n_stoch)
    between = y.mean(1).var(ddof=1) - y.var(1, ddof=1).mean() / n_stoch
    total = y.var(ddof=1); return float(max(between, 0) / total), float(total)

def summarise(x):
    return dict(median=float(np.median(x)), mean=float(np.mean(x)),
                lo95=float(np.percentile(x, 2.5)), hi95=float(np.percentile(x, 97.5)))

def prcc(X, y):
    """Partial rank correlation coefficients (Marino et al. 2008)."""
    R = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])]); ry = rankdata(y)
    out = []
    for j in range(R.shape[1]):
        A = np.column_stack([np.ones(len(ry)), np.delete(R, j, axis=1)])
        r1 = R[:, j] - A @ np.linalg.lstsq(A, R[:, j], rcond=None)[0]
        r2 = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
        out.append(float(np.corrcoef(r1, r2)[0, 1]))
    return np.array(out)

# ----------------------------------------------------------------------------------
# validation: crude Bernoulli event simulator vs closed form (frozen environment)
# ----------------------------------------------------------------------------------
def crude_event_mc(P, S, W, comp, n, rng):
    lam = float(lam_attempt(P, S)); ma = float(m_admin(P, S, P.interlock)); mis = 0
    attempt = rng.random(n) < lam; na = int(attempt.sum())
    if P.interlock:
        pi = float(p_interlock_pass(P, S, W, comp)); cc = rng.random(na) < P.beta_cc
        u = rng.random(na); v = rng.random(na); w = rng.random(na)
        fail_cc = cc & (u < min(pi, ma))
        fail_ind = (~cc) & (v < pi) & (w < ma)
        mis += int(fail_cc.sum() + fail_ind.sum())
        sens = P.d_sens * (not comp)
    else:
        mis += int((rng.random(na) < ma).sum()); sens = 0.0
    load = rng.random(n) < P.p_load
    mis += int((load & (rng.random(n) < P.m_load_adm * (1 - sens))).sum())
    return mis / n

# ----------------------------------------------------------------------------------
# fail-safe permissive logic (hardware/firmware contract; used for Table 1 + tests)
# ----------------------------------------------------------------------------------
def permissive(tag_valid: bool, record_found: bool, product_match: bool, operator_auth: bool,
               heartbeat_ok: bool, override_token: bool = False, supervisor_2fa: bool = False) -> bool:
    """Relay energised (fuel flow permitted) only if every condition is affirmatively true.
    Any False / missing input -> de-energised (normally-open, fail-safe)."""
    normal = tag_valid and record_found and product_match
    override = (not tag_valid) and override_token and supervisor_2fa
    return bool(heartbeat_ok and operator_auth and (normal or override))

# ----------------------------------------------------------------------------------
# pilot verification statistics
# ----------------------------------------------------------------------------------
def drills_required(p_miss_claim, conf=0.95):
    """Seeded-mismatch drills with zero failures needed to demonstrate miss prob <= claim."""
    return math.ceil(math.log(1 - conf) / math.log(1 - p_miss_claim))

def p_observe_near_miss(P, trucks, days):
    lam = float(lam_attempt(P, P.S_a / (P.S_a + P.S_b)))
    mu = trucks * P.events_per_truck_day * days * lam
    return 1 - math.exp(-mu), mu

# ----------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------
C = dict(k="#222222", g="#7f7f7f", b="#1f4e79", o="#c55a11", r="#a61e1e", n="#2e7d32")
def _fig_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.color": "#dddddd",
                         "font.family": "DejaVu Serif", "figure.dpi": 150})
    return plt

def fig_architecture(path):
    plt = _fig_style(); from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    f, ax = plt.subplots(figsize=(7.2, 3.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.grid(False)
    def box(x, y, w, h, t, fc="#f2f2f2", ec="#222222", fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
    def arr(a, b, t=None):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=1, color="#222222"))
        if t: ax.text((a[0] + b[0]) / 2 + (0.55 if a[0] == b[0] else 0), (a[1] + b[1]) / 2 + (0 if a[0] == b[0] else 0.15), t, fontsize=7, ha="center")
    box(0.1, 3.5, 2.2, 1.1, "Aircraft-side tag\n(2-D code, per fuel port,\nbound to registration)")
    box(0.1, 1.9, 2.2, 1.1, "Truck-side laser\nscanner / rugged\nhandheld")
    box(0.1, 0.2, 2.2, 1.1, "Operator ID\n(RFID / PIN)")
    box(3.3, 1.9, 2.6, 2.7, "Edge controller\nparse tag \u2192 registration\nlookup fuel requirement\n(manifest, STC data)\ncompare with truck product\nlog (hash-chained)", fc="#dde8f3")
    box(3.3, 0.2, 2.6, 1.1, "Supervisor override\n(2-factor, logged)", fc="#fbe9dd")
    box(6.9, 3.4, 2.9, 1.2, "Fail-safe relay\n(normally open; in series\nwith deadman / ESD solenoid)", fc="#e3f1e4")
    box(6.9, 1.9, 2.9, 1.1, "Fuel flow permitted\nonly if relay energised")
    box(6.9, 0.2, 2.9, 1.1, "Optional: inline product\nverification sensor", fc="#fff6d6")
    arr((1.2, 3.5), (1.2, 3.0), "optical read"); arr((2.3, 2.45), (3.3, 3.0), "code")
    arr((2.3, 0.75), (3.3, 2.2), "auth"); arr((4.6, 1.3), (4.6, 1.9)); arr((5.9, 3.9), (6.9, 4.0), "permit")
    arr((8.35, 3.4), (8.35, 3.0)); arr((8.35, 1.3), (8.35, 1.9)); ax.text(8.5, 1.6, "veto", fontsize=7)
    f.tight_layout(); f.savefig(path, dpi=200, bbox_inches="tight"); plt.close(f)

def fig_hazard_bars(res, path):
    plt = _fig_style(); f, ax = plt.subplots(figsize=(6.2, 3.6)); keys = list(res)
    med = [res[k]["pmfe"]["median"] for k in keys]; lo = [res[k]["pmfe"]["lo95"] for k in keys]
    hi = [res[k]["pmfe"]["hi95"] for k in keys]; x = np.arange(len(keys))
    ax.bar(x, med, color=[C["g"], C["o"], C["b"], C["n"]][:len(keys)], edgecolor="k", zorder=3)
    ax.errorbar(x, med, yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
                fmt="none", ecolor="k", capsize=4, zorder=4)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([k.split("_")[0] for k in keys])
    ax.set_ylabel("Misfuelling events per $10^6$ fuellings"); ax.set_xlabel("Barrier configuration")
    f.tight_layout(); f.savefig(path, dpi=200); plt.close(f)

def fig_bifurcation(P, path, mc_points=None):
    plt = _fig_style(); mf = MeanField(P); win = mf.bistable_window(); rr = np.linspace(0.3, 20, 500)
    hi, lo = [], []
    for rho in rr:
        st = [r for r in mf.fixed_points(rho) if mf.stable(r, rho)]
        hi.append(max(st) if st else np.nan); lo.append(min(st) if len(st) > 1 else np.nan)
    f, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(rr, hi, color=C["r"], lw=2, label="stable branches"); ax.plot(rr, lo, color=C["r"], lw=2)
    if win: ax.axvspan(*win, color=C["o"], alpha=0.15, label=f"bistable window  {win[0]:.1f}\u2013{win[1]:.1f}")
    if mc_points is not None: ax.scatter(*mc_points, s=10, color=C["b"], alpha=0.4, label="fleet MC, year-mean (S1)", zorder=5)
    ax.set_xlabel(r"Control group $\rho = R/\lambda_{\rm eff}$"); ax.set_ylabel("Compromised fraction $I^*$")
    ax.set_ylim(-0.02, 1.02); ax.set_xlim(0, 20); ax.legend(fontsize=7); f.tight_layout(); f.savefig(path, dpi=200); plt.close(f)
    return win

def fig_shock(series_by_label, path, shock):
    plt = _fig_style(); f, ax = plt.subplots(figsize=(6.4, 3.6)); cols = [C["o"], C["b"], C["n"]]
    for (lab, s), c, ls in zip(series_by_label.items(), cols, ["-", "--", ":"]):
        t = np.arange(s.shape[1]); ax.plot(t, np.median(s, 0), color=c, lw=1.8, ls=ls, label=lab)
        ax.fill_between(t, np.percentile(s, 25, 0), np.percentile(s, 75, 0), color=c, alpha=0.2)
    ax.axvspan(shock[0], shock[1], color=C["g"], alpha=0.15, label="tag-fouling shock")
    ax.set_xlabel("Day of operation"); ax.set_ylabel("Compromised fraction of fleet"); ax.legend(fontsize=7)
    f.tight_layout(); f.savefig(path, dpi=200); plt.close(f)

def fig_prcc(names, vals, path):
    plt = _fig_style(); o = np.argsort(np.abs(vals)); f, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.barh(np.array(names)[o], np.array(vals)[o], color=[C["b"] if v > 0 else C["o"] for v in np.array(vals)[o]], edgecolor="k")
    ax.set_xlabel("PRCC with misfuelling rate (S1)"); f.tight_layout(); f.savefig(path, dpi=200); plt.close(f)

def fig_ops(res, path):
    plt = _fig_style(); f, ax = plt.subplots(1, 2, figsize=(7.0, 3.2)); keys = [k for k in res if k != "S0_baseline_admin"]
    x = np.arange(len(keys)); lab = [k.split("_")[0] for k in keys]
    ax[0].bar(x, [100 * res[k]["nuis_rate"]["median"] for k in keys], color=C["b"], edgecolor="k", zorder=3)
    ax[0].set_xticks(x); ax[0].set_xticklabels(lab); ax[0].set_ylabel("Nuisance interlocks (% of fuellings)")
    ax[1].bar(x, [res[k]["delay_s"]["median"] for k in keys], color=C["o"], edgecolor="k", zorder=3)
    ax[1].set_xticks(x); ax[1].set_xticklabels(lab); ax[1].set_ylabel("Mean added delay (s per fuelling)")
    f.tight_layout(); f.savefig(path, dpi=200); plt.close(f)

def fig_pilot(P, path):
    plt = _fig_style(); f, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for tr, ls in [(5, "-"), (20, "--"), (60, ":")]:
        d = np.arange(10, 721, 10); ax[0].plot(d, [p_observe_near_miss(P, tr, x)[0] for x in d], ls, color=C["k"], label=f"{tr} trucks")
    ax[0].set_xlabel("Pilot duration (days)"); ax[0].set_ylabel("P(\u2265 1 real mismatch attempt)"); ax[0].legend(fontsize=7)
    c = np.logspace(-3, -1, 100); ax[1].plot(c, [drills_required(x) for x in c], color=C["r"], lw=2)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("Claimed interlock pass probability (miss)")
    ax[1].set_ylabel("Seeded drills, zero failures (95 %)"); f.tight_layout(); f.savefig(path, dpi=200); plt.close(f)

# ==================================================================================
# v2 additions: state machine, fault injection, dependence/sensor/phase studies, economics
# ==================================================================================
# ---------- controller state machine (start permission vs continuous-flow permission) ----------
STATES = ["IDLE", "VERIFY", "AUTH_DELIVERY", "FAULT", "CONTROLLED_SHUTDOWN", "RECOVERY"]
TRANSITIONS = {
    ("IDLE", "tag_scanned"): "VERIFY",
    ("VERIFY", "verify_ok"): "AUTH_DELIVERY",          # only edge that energises the start permissive
    ("VERIFY", "verify_fail"): "IDLE", ("VERIFY", "timeout"): "IDLE", ("VERIFY", "fault"): "FAULT",
    ("AUTH_DELIVERY", "delivery_complete"): "IDLE",     # deadman release / normal end
    ("AUTH_DELIVERY", "esd_trip"): "IDLE",              # independent ESD path; controller only observes
    ("AUTH_DELIVERY", "product_veto"): "CONTROLLED_SHUTDOWN",
    ("AUTH_DELIVERY", "record_revoked"): "CONTROLLED_SHUTDOWN",
    ("AUTH_DELIVERY", "controller_fault"): "FAULT",     # flow completes under deadman; new starts inhibited
    ("CONTROLLED_SHUTDOWN", "flow_stopped"): "RECOVERY",
    ("FAULT", "supervisor_reset"): "RECOVERY",
    ("RECOVERY", "self_test_ok"): "IDLE", ("RECOVERY", "self_test_fail"): "FAULT",
}
FAULT_CLASSES = {   # response of the interlock to each fault, by class
    "A_start_inhibit": ["tag invalid", "record missing/stale", "product mismatch", "operator not authenticated",
                        "heartbeat/self-test failure before start", "relay contact feedback mismatch"],
    "B_controlled_shutdown": ["product-sensor veto during flow", "aircraft record revoked during flow"],
    "C_alarm_and_complete": ["controller or comms fault during flow (start permit is sealed in; next start inhibited)"],
    "D_independent_ESD": ["deadman release", "emergency shut-off trip (not routed through the controller)"],
}
class StateMachine:
    def __init__(self): self.state = "IDLE"; self.log = []
    def fire(self, ev):
        nxt = TRANSITIONS.get((self.state, ev)); self.log.append((self.state, ev, nxt or self.state))
        if nxt: self.state = nxt
        return self.state
    @staticmethod
    def start_permit_energised(state): return state == "AUTH_DELIVERY"

def reachable_without(target, avoid, start="IDLE"):
    """Is `target` reachable from `start` without visiting `avoid`?"""
    seen = {start}; stack = [start]
    while stack:
        u = stack.pop()
        for (a, _), b in TRANSITIONS.items():
            if a == u and b not in seen and b != avoid: seen.add(b); stack.append(b)
    return target in seen

def surge_pressure(Q_lpm=2000.0, d_mm=75.0, rho=800.0, a=400.0, L=15.0, t_close=None):
    """Joukowsky surge (illustrative; hose wave speed a is an ASSUMED value). Returns dict (Pa)."""
    A = math.pi * (d_mm / 1000) ** 2 / 4; v = (Q_lpm / 1000 / 60) / A; dP_full = rho * a * v
    t_crit = 2 * L / a
    dP = dP_full if (t_close is None or t_close <= t_crit) else dP_full * t_crit / t_close
    return dict(v=v, dP_instant=dP_full, t_crit=t_crit, dP=dP)

# ---------- virtual fault-injection campaign on the controller logic ----------
@dataclass
class CtrlCfg:
    stale_limit: bool = True      # reject records older than the age limit (covers comms loss)
    relay_feedback: bool = True   # mirror-contact feedback detects welded relay at start
    two_factor: bool = True       # override needs token AND supervisor factor
    nonce_expiry: bool = True     # override tokens single-use and time-limited (replay/expired)
    stand_check: bool = True      # GPS/stand association vs manifest (wrong-aircraft scan)
    diverse_check: bool = True    # independent second comparison channel (1oo2D) for software fault
    registry_xcheck: bool = True  # second-source cross-check of the fuel requirement (STC/registry flags)
    sensor_delta: float = 0.95    # inline product sensor detection probability (0 = no sensor)
    d_feedback: float = 0.99; d_stand: float = 0.90; d_diverse: float = 0.95; d_xcheck: float = 0.80

FI_CATEGORIES = ["wrong_record", "stale_record", "comms_loss", "wrong_aircraft_scan", "sensor_misclassification",
                 "relay_weld", "software_stuck_permit", "power_interruption", "override_attack"]
def inject(cat, cfg, rng, n):
    """Return boolean array: True where an injected *dangerous* scenario (true product mismatch present)
    still yields a start permit. Scenario sub-types inside a category are sampled uniformly."""
    u = rng.random(n)
    if cat == "wrong_record":            # record states a compatible requirement but it is wrong
        return u >= (cfg.d_xcheck if cfg.registry_xcheck else 0.0)
    if cat == "stale_record":            # record out of date (e.g. after STC conversion)
        return np.full(n, not cfg.stale_limit) & (u >= 0)
    if cat == "comms_loss":              # cache used beyond age limit
        return np.full(n, not cfg.stale_limit) & (u >= 0)
    if cat == "wrong_aircraft_scan":     # adjacent aircraft scanned
        return u >= (cfg.d_stand if cfg.stand_check else 0.0)
    if cat == "sensor_misclassification":
        return u >= cfg.sensor_delta
    if cat == "relay_weld":              # welded-closed relay: permit is present regardless of logic
        return u >= (cfg.d_feedback if cfg.relay_feedback else 0.0)
    if cat == "software_stuck_permit":
        return u >= (cfg.d_diverse if cfg.diverse_check else 0.0)
    if cat == "power_interruption":      # relay de-energises: always safe
        return np.zeros(n, bool)
    if cat == "override_attack":         # token only / 2FA only / replay / expired
        sub = rng.integers(0, 4, n); ok = np.zeros(n, bool)
        ok |= (sub == 0) & (not cfg.two_factor); ok |= (sub == 1) & (not cfg.two_factor)
        ok |= (sub == 2) & (not cfg.nonce_expiry); ok |= (sub == 3) & (not cfg.nonce_expiry)
        return ok
    raise ValueError(cat)

def fault_injection_campaign(n_per_cat=20000, seed=0):
    rng = np.random.default_rng(seed); out = {}
    cfgs = {"full": CtrlCfg(), "no_sensor": CtrlCfg(sensor_delta=0.0)}
    abl = {"none (all defences off)": CtrlCfg(False, False, False, False, False, False, False, 0.0),
           "full": CtrlCfg()}
    for name in ("stale_limit", "relay_feedback", "two_factor", "nonce_expiry", "stand_check", "diverse_check", "registry_xcheck"):
        abl["full minus " + name] = CtrlCfg(**{name: False})
    abl["full minus sensor"] = CtrlCfg(sensor_delta=0.0)
    for cname, cfg in abl.items():
        cats = {}; tot = 0; bad = 0
        for c in FI_CATEGORIES:
            r = inject(c, cfg, rng, n_per_cat); k = int(r.sum()); cats[c] = dict(unsafe=k, n=n_per_cat, p=k / n_per_cat, ci=wilson(k, n_per_cat))
            tot += n_per_cat; bad += k
        out[cname] = dict(categories=cats, unsafe=bad, n=tot, p=bad / tot, ci=wilson(bad, tot))
    return out

# ---------- dependence, sensor, phase-diagram and economics studies ----------
def dependence_grid(betas=(0.0, 0.05, 0.2, 0.5, 1.0), comps=(1.0, 3.0, 10.0)):
    out = {}
    for k in ("S1_interlock_basic", "S3_interlock_full"):
        g = {}
        for b in betas:
            for c in comps:
                h = nominal_hazard(PRESETS[k].copy(beta_cc=b, complacency=c)); h0 = nominal_hazard(PRESETS["S0_baseline_admin"])
                g[f"{b}|{c}"] = dict(h=1e6 * h, rrr=1 - h / h0)
        out[k] = g
    return out

def sensor_sweep(deltas=(0.0, 0.5, 0.8, 0.9, 0.95, 0.99)):
    h0 = nominal_hazard(PRESETS["S0_baseline_admin"]); out = {}
    for d in deltas:
        P = PRESETS["S3_interlock_full"].copy(d_sens=d); h = nominal_hazard(P)
        mf = MeanField(P); out[str(d)] = dict(h=1e6 * h, rrr=1 - h / h0, nuis=float(p_nuisance(P, P.W_mean)), rho=mf.rho())
    return out

def phase_bistable_width(betas, hs, K=0.5):
    Wd = np.zeros((len(hs), len(betas)))
    for i, h in enumerate(hs):
        for j, b in enumerate(betas):
            w = MeanField(Params(beta_c=b, h_c=h, K_c=K)).bistable_window(); Wd[i, j] = (w[1] - w[0]) if w else 0.0
    return Wd

def phase_collapse_map(betas, tds, n_rep=400, seed=11):
    shock = (120, 180, 4.0); M = np.zeros((len(tds), len(betas)))
    for i, td in enumerate(tds):
        for j, b in enumerate(betas):
            e = run_ensemble(PRESETS["S1_interlock_basic"].copy(n_trucks=20, beta_c=b, T_detect=td), n_rep, seed, shock=shock)
            M[i, j] = e["collapse"].mean()
    return M

def annual_cost(kind, N, level="nom"):
    """Annualised cost per vehicle (USD). Hardware/software figures from the concept; S3 hardware is an ASSUMPTION."""
    i = {"low": 0, "nom": 1, "high": 2}[level]
    hw = {"S1": (1200, 1500, 1800), "S3": (5000, 8000, 12000)}[kind][i]
    host = (50, 75, 100)[i] * 12; sw = (35000, 40000, 45000)[i] / (N * 5.0)
    return hw / 7.0 + 0.10 * hw + host + sw + (0.0 if kind == "S1" else 150.0)   # 10 %/yr install+maintenance; S3 calibration

def breakeven_event_cost(kind, N, E, f=1.0, level="nom"):
    P0 = PRESETS["S0_baseline_admin"].copy(lam0=PRESETS["S0_baseline_admin"].lam0 * f, p_load=PRESETS["S0_baseline_admin"].p_load * f)
    key = {"S1": "S1_interlock_basic", "S3": "S3_interlock_full"}[kind]
    P1 = PRESETS[key].copy(lam0=PRESETS[key].lam0 * f, p_load=PRESETS[key].p_load * f)
    dh = nominal_hazard(P0) - nominal_hazard(P1)
    return annual_cost(kind, N, level) / (dh * E), dh

# ---------- figures (v2) ----------
def fig_statemachine(path):
    plt = _fig_style(); from matplotlib.patches import Ellipse, FancyArrowPatch
    f, ax = plt.subplots(figsize=(7.4, 4.0)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.grid(False)
    pos = {"IDLE": (1.2, 3.8), "VERIFY": (4.0, 3.8), "AUTH_DELIVERY": (7.2, 3.8), "FAULT": (4.0, 1.2),
           "CONTROLLED_SHUTDOWN": (8.3, 1.2), "RECOVERY": (1.2, 1.2)}
    col = {"AUTH_DELIVERY": "#e3f1e4", "FAULT": "#f6d5d5", "CONTROLLED_SHUTDOWN": "#fbe9dd", "RECOVERY": "#fff6d6"}
    for n, (x, y) in pos.items():
        ax.add_patch(Ellipse((x, y), 2.2, 1.0, fc=col.get(n, "#f2f2f2"), ec="#222", lw=1)); ax.text(x, y, n.replace("_", "\n") if len(n) > 8 else n, ha="center", va="center", fontsize=7)
    def arr(a, b, t, rad=0.0, off=(0, 0.18)):
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1, color="#222", connectionstyle=f"arc3,rad={rad}", shrinkA=26, shrinkB=26))
        ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], t, fontsize=6.5, ha="center")
    arr("IDLE", "VERIFY", "tag_scanned", off=(0, -0.3)); arr("VERIFY", "AUTH_DELIVERY", "verify_ok\n(start permit)")
    arr("VERIFY", "IDLE", "verify_fail / timeout", rad=0.45, off=(0, 0.95))
    arr("AUTH_DELIVERY", "IDLE", "delivery_complete / esd_trip", rad=-0.45, off=(0, 1.45))
    arr("AUTH_DELIVERY", "CONTROLLED_SHUTDOWN", "product_veto /\nrecord_revoked", off=(0.9, 0))
    arr("AUTH_DELIVERY", "FAULT", "controller_fault\n(flow completes\nunder deadman)", off=(0.3, 0.5), rad=0.15)
    arr("VERIFY", "FAULT", "fault", off=(0.3, 0)); arr("CONTROLLED_SHUTDOWN", "RECOVERY", "flow_stopped", rad=-0.2, off=(0, -0.7))
    arr("FAULT", "RECOVERY", "supervisor_reset", off=(0, 0.2)); arr("RECOVERY", "IDLE", "self_test_ok", off=(-0.6, 0))
    f.tight_layout(); f.savefig(path, dpi=200, bbox_inches="tight"); plt.close(f)

def fig_faulttree(path):
    plt = _fig_style(); from matplotlib.patches import FancyBboxPatch, Circle
    f, ax = plt.subplots(figsize=(7.4, 4.6)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.grid(False)
    def box(x, y, w, t, fc="#f2f2f2"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.3), w, 0.6, boxstyle="round,pad=0.03", fc=fc, ec="#222", lw=1)); ax.text(x, y, t, ha="center", va="center", fontsize=6.5)
    def gate(x, y, t): ax.add_patch(Circle((x, y), 0.28, fc="white", ec="#222")); ax.text(x, y, t, ha="center", va="center", fontsize=6)
    def line(a, b): ax.plot([a[0], b[0]], [a[1], b[1]], color="#222", lw=0.8)
    box(6, 6.5, 3.6, "Uncorrected wrong-product delivery", "#f6d5d5"); gate(6, 5.8, "OR")
    box(3, 5.0, 3.4, "Path 1: wrong-product attempt, not stopped"); box(9.5, 5.0, 3.4, "Path 2: product differs from record"); line((6, 6.2), (6, 6.08)); line((6, 5.52), (3, 5.3)); line((6, 5.52), (9.5, 5.3))
    gate(3, 4.3, "AND"); line((3, 4.7), (3, 4.58))
    box(1.0, 3.4, 1.8, "Attempt λ(S)"); box(3.0, 3.4, 1.8, "Admin check misses m(S)"); box(5.2, 3.4, 1.9, "Interlock permits π", "#dde8f3")
    for x in (1.0, 3.0, 5.2): line((3, 4.02), (x, 3.7))
    gate(5.2, 2.7, "OR"); line((5.2, 3.1), (5.2, 2.98))
    leaves = [(0.9, "Bypassed /\ncompromised\n(c = 1)"), (2.9, "Override\npassed\n(p_u m_o)"), (4.9, "Wrong\nassociation\n(p_a φ)"), (6.9, "Relay DU /\nsoftware\nfault")]
    for x, t in leaves:
        ax.add_patch(FancyBboxPatch((x - 0.85, 0.9), 1.7, 1.0, boxstyle="round,pad=0.03", fc="#fff6d6", ec="#222", lw=1)); ax.text(x, 1.4, t, ha="center", va="center", fontsize=6.5); line((5.2, 2.42), (x, 1.9))
    gate(9.5, 4.3, "AND"); line((9.5, 4.7), (9.5, 4.58))
    box(8.3, 3.4, 2.0, "Loading error p_L"); box(10.7, 3.4, 2.0, "Admin misses m_L"); box(9.5, 2.5, 2.6, "Product sensor misses (1−δ)", "#e3f1e4")
    line((9.5, 4.02), (8.3, 3.7)); line((9.5, 4.02), (10.7, 3.7)); line((9.5, 4.02), (9.5, 2.8))
    f.tight_layout(); f.savefig(path, dpi=200, bbox_inches="tight"); plt.close(f)

def fig_convergence(ens, path):
    plt = _fig_style(); f, ax = plt.subplots(1, 2, figsize=(7.2, 3.2)); n = len(ens["pmfe"]); idx = np.unique(np.logspace(1.5, math.log10(n), 40).astype(int))
    med = [np.median(ens["pmfe"][:i]) for i in idx]; ax[0].plot(idx, med, color=C["b"]); ax[0].set_xscale("log"); ax[0].set_xlabel("Replicates"); ax[0].set_ylabel("Running median PMFE (S1)")
    pc = [ens["collapse"][:i].mean() for i in idx]; ci = [wilson(int(ens["collapse"][:i].sum()), i) for i in idx]
    ax[1].plot(idx, pc, color=C["r"]); ax[1].fill_between(idx, [c[0] for c in ci], [c[1] for c in ci], color=C["r"], alpha=0.2)
    ax[1].set_xscale("log"); ax[1].set_xlabel("Replicates"); ax[1].set_ylabel("Collapse probability (Wilson 95 %)")
    f.tight_layout(); f.savefig(path, dpi=200); plt.close(f)

def fig_phase(betas, hs, Wd, bt, tds, M, path, rho_hi_fn, lam_eff):
    plt = _fig_style(); f, ax = plt.subplots(1, 2, figsize=(7.6, 3.4))
    im = ax[0].pcolormesh(betas, hs, Wd, shading="auto", cmap="Greys"); ax[0].set_xlabel(r"Peer-feedback strength $\beta_c$"); ax[0].set_ylabel(r"Feedback exponent $h$")
    ax[0].plot([20], [6], "o", color=C["r"], mfc="none", ms=9, label="MyUncle defaults"); ax[0].legend(fontsize=7, loc="lower right"); f.colorbar(im, ax=ax[0], label=r"Window width $\Delta\rho$")
    im2 = ax[1].pcolormesh(bt, tds, M, shading="auto", cmap="Greys", vmin=0, vmax=max(0.35, float(np.max(M))))
    ax[1].set_xlabel(r"Peer-feedback strength $\beta_c$"); ax[1].set_ylabel(r"Detect/restore time $T_d$ (days)"); ax[1].set_yscale("log")
    bb = np.linspace(min(bt), max(bt), 60); td_star = [1.0 / (lam_eff * rho_hi_fn(b)) if rho_hi_fn(b) else np.nan for b in bb]
    ax[1].plot(bb, td_star, "--", color=C["r"], lw=1.5, label=r"$\rho=\rho_{\rm hi}(\beta_c)$"); ax[1].legend(fontsize=7, loc="upper right"); ax[1].set_ylim(min(tds), max(tds))
    f.colorbar(im2, ax=ax[1], label="Collapse probability under shock"); f.tight_layout(); f.savefig(path, dpi=200); plt.close(f)

def fig_econ(path):
    plt = _fig_style(); f, ax = plt.subplots(1, 2, figsize=(7.4, 3.3)); fs = np.logspace(-0.7, 0.7, 40); Es = np.linspace(2000, 16000, 40)
    for a, kind in zip(ax, ("S1", "S3")):
        Z = np.array([[math.log10(breakeven_event_cost(kind, 60, E, f_)[0]) for f_ in fs] for E in Es])
        cs = a.contourf(fs, Es, Z, levels=np.arange(3.5, 8.6, 0.5), cmap="Greys"); a.set_xscale("log")
        a.contour(fs, Es, Z, levels=[5], colors=[C["r"]], linewidths=1.6); a.set_title(kind + " (60 vehicles)", fontsize=9)
        a.set_xlabel("Error-propensity multiplier $f$"); a.set_ylabel("Fuellings per vehicle-year")
    f.colorbar(cs, ax=ax, label=r"$\log_{10}$ break-even cost per unsafe delivery (US\$)"); f.savefig(path, dpi=200, bbox_inches="tight"); plt.close(f)

def fig_dependence(dep, path):
    plt = _fig_style(); f, ax = plt.subplots(figsize=(5.6, 3.3)); comps = (1.0, 3.0, 10.0); betas = (0.0, 0.05, 0.2, 0.5, 1.0)
    for k, ls in (("S1_interlock_basic", "-"), ("S3_interlock_full", "--")):
        for c, col in zip(comps, (C["b"], C["o"], C["r"])):
            ax.plot(range(len(betas)), [100 * dep[k][f"{b}|{c}"]["rrr"] for b in betas], ls, marker="o", ms=3, color=col, label=f"{k[:2]}, complacency ×{c:g}")
    ax.set_xticks(range(len(betas))); ax.set_xticklabels([f"{b:g}" for b in betas]); ax.set_xlabel(r"Common-cause beta factor $\beta_{cc}$ (0 = independent, 1 = fully dependent)")
    ax.set_ylabel("Nominal hazard reduction vs S0 (%)"); ax.legend(fontsize=6, ncol=2); ax.set_ylim(60, 100)
    f.tight_layout(); f.savefig(path, dpi=200); plt.close(f)

# ---------- reproduce v2 ----------
def reproduce2(outdir="misfuel_out_v2", seed=7, n_par=1000, n_stoch=5, verbose=True):
    import matplotlib, sys, platform, scipy; matplotlib.use("Agg"); os.makedirs(outdir, exist_ok=True); fig = lambda n: os.path.join(outdir, n)
    n_rep = n_par * n_stoch; R = {"meta": dict(seed=seed, n_par=n_par, n_stoch=n_stoch, n_rep=n_rep, python=sys.version.split()[0],
        numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__, platform=platform.platform(),
        collapse_definition="year-mean compromised fraction of the fleet > 0.5",
        scenarios={"pilot": "5 vehicles, 365 d, no shock", "regional": "60 vehicles, 365 d, no shock",
                   "shock": "20 vehicles, 365 d, 60-d tag-fouling shock (x4 unreadable probability) from day 120"})}
    fig_architecture(fig("fig1_architecture.png")); fig_statemachine(fig("fig2_statemachine.png")); fig_faulttree(fig("fig3_faulttree.png"))
    R["breakdown_per_1e6"] = {k: hazard_breakdown(P) for k, P in PRESETS.items()}
    R["nominal_hazard_per_1e6"] = {k: 1e6 * nominal_hazard(P) for k, P in PRESETS.items()}
    # --- ensembles
    ens = {k: run_ensemble(P, n_rep, seed, n_stoch=n_stoch) for k, P in PRESETS.items()}
    base = ens["S0_baseline_admin"]["pmfe"]; res = {}
    for k, e in ens.items():
        res[k] = {m: summarise(e[m]) for m in ("pmfe", "exp_mis", "nuis_rate", "comp_mean", "delay_s", "p_delay5")}
        kk = int(e["collapse"].sum()); res[k]["collapse"] = dict(p=kk / n_rep, ci=wilson(kk, n_rep), n=n_rep)
        res[k]["rrr_paired"] = summarise(1 - e["pmfe"] / base)
        res[k]["var_share_parameters"], res[k]["var_log10_total"] = variance_decomposition(e["pmfe"], n_par, n_stoch)
        if verbose: print(k, round(res[k]["pmfe"]["median"], 3), res[k]["collapse"])
    R["pilot_fleet"] = res
    reg = {}
    for k in PRESETS:
        e = run_ensemble(PRESETS[k].copy(n_trucks=60), 2000, seed + 1, n_stoch=5)
        kk = int(e["collapse"].sum()); reg[k] = {m: summarise(e[m]) for m in ("pmfe", "exp_mis", "comp_mean")}
        reg[k]["collapse"] = dict(p=kk / len(e["collapse"]), ci=wilson(kk, len(e["collapse"])), n=len(e["collapse"]))
    R["regional_fleet"] = reg
    shock = (120, 180, 4.0); shk = {}; ser = {}
    for k in ("S1_interlock_basic", "S2_interlock_audit", "S3_interlock_full"):
        e = run_ensemble(PRESETS[k].copy(n_trucks=20), 5000, seed + 3, shock=shock, n_stoch=5, keep_series=True)
        kk = int(e["collapse"].sum()); shk[k] = dict(collapse=dict(p=kk / len(e["collapse"]), ci=wilson(kk, len(e["collapse"])), n=len(e["collapse"])),
                                                    pmfe=summarise(e["pmfe"]), comp_end=summarise(e["comp_end"]))
        ser[PRESETS[k].label] = e["series"][::25]
    R["shock"] = shk
    # attribution of S1 shock collapse: peer feedback vs slow restoration
    attr = {}
    for nm, kw in {"S1 as specified": {}, "no peer feedback (beta_c = 0)": dict(beta_c=0.0), "T_d = 7 d, feedback on": dict(T_detect=7.0),
                   "no feedback and T_d = 7 d": dict(beta_c=0.0, T_detect=7.0)}.items():
        e = run_ensemble(PRESETS["S1_interlock_basic"].copy(n_trucks=20, **kw), 5000, seed + 3, shock=shock, n_stoch=5)
        kk = int(e["collapse"].sum()); attr[nm] = dict(p=kk / len(e["collapse"]), ci=wilson(kk, len(e["collapse"])), comp_mean=float(e["comp_mean"].mean()))
    R["shock_attribution"] = attr
    # --- figures: hazard bars, bifurcation, shock, prcc, ops, pilot, convergence
    fig_hazard_bars(res, fig("fig4_hazard.png"))
    xs, ys = [], []; Zs = np.random.default_rng(seed + 5).normal(size=(150, len(PRIORS)))
    Pb = draw_params_batch(PRESETS["S1_interlock_basic"].copy(n_trucks=60), Zs); sim = simulate_batch(Pb, 150, np.random.default_rng(seed + 6))
    for i in range(150):
        Pi = draw_params(PRESETS["S1_interlock_basic"], Zs[i]); xs.append(MeanField(Pi).rho())
    R["bistable_window"] = fig_bifurcation(PRESETS["S1_interlock_basic"], fig("fig5_bifurcation.png"), (xs, sim["comp_mean"]))
    R["rho_nominal"] = {k: MeanField(P).rho() for k, P in PRESETS.items() if P.interlock}
    R["lam_eff_per_day_nominal"] = {k: MeanField(P).lam_eff() for k, P in PRESETS.items() if P.interlock}
    fig_shock(ser, fig("fig6_shock.png"), shock)
    names = list(PRIORS); pr = prcc(ens["S1_interlock_basic"]["params"], ens["S1_interlock_basic"]["pmfe"])
    R["prcc_S1"] = dict(zip(names, map(float, pr))); fig_prcc(names, pr, fig("fig7_prcc.png"))
    fig_ops(res, fig("fig8_ops.png")); fig_pilot(PRESETS["S1_interlock_basic"], fig("fig9_pilot.png"))
    fig_convergence(ens["S1_interlock_basic"], fig("fig10_convergence.png"))
    P1 = PRESETS["S1_interlock_basic"]
    R["pilot"] = dict(p_obs_5trucks_120d=p_observe_near_miss(P1, 5, 120), drills_miss_1e2=drills_required(1e-2), drills_miss_1e3=drills_required(1e-3))
    # --- prior robustness
    rob = {}
    for nm, kw in {"lognormal (baseline)": {}, "wider priors (GSD^1.5)": dict(gsd_scale=1.5), "log-uniform, same SD": dict(prior="loguniform")}.items():
        r = {}
        for k in ("S0_baseline_admin", "S1_interlock_basic", "S3_interlock_full"):
            e = run_ensemble(PRESETS[k], 2500, seed + 7, n_stoch=5, **kw); r[k] = dict(pmfe=summarise(e["pmfe"]), collapse=float(e["collapse"].mean()))
            if k == "S0_baseline_admin": b0 = e["pmfe"]
            else: r[k]["rrr"] = summarise(1 - e["pmfe"] / b0)
        rob[nm] = r
    R["prior_robustness"] = rob
    # --- dependence, sensor, phase diagram
    R["dependence"] = dependence_grid(); fig_dependence(R["dependence"], fig("fig11_dependence.png")); R["sensor_sweep"] = sensor_sweep()
    betas = np.array([2, 4, 6, 8, 10, 14, 20, 30, 40.0]); hs = np.array([1, 2, 3, 4, 6, 8, 10.0]); Wd = phase_bistable_width(betas, hs)
    bt = np.array([4, 8, 12, 20, 30, 40.0]); tds = np.array([2, 4, 7, 14, 30, 60, 90, 180.0]); M = phase_collapse_map(bt, tds)
    lam_eff = MeanField(P1).lam_eff()
    def rho_hi(b):
        w = MeanField(Params(beta_c=b)).bistable_window(); return w[1] if w else 0.0
    fig_phase(betas, hs, Wd, bt, tds, M, fig("fig12_phase.png"), lambda b: rho_hi(b) or np.nan, lam_eff)
    R["phase"] = dict(betas=betas.tolist(), hs=hs.tolist(), width=Wd.tolist(), bt=bt.tolist(), tds=tds.tolist(), collapse=M.tolist(),
                      frac_bistable_cells=float((Wd > 0).mean()), min_beta_bistable_h6=float(min([b for b, w in zip(betas, Wd[hs.tolist().index(6.0)]) if w > 0] or [np.nan])))
    # --- fault injection
    R["fault_injection"] = fault_injection_campaign(20000, seed + 8)
    R["state_machine"] = dict(states=STATES, n_transitions=len(TRANSITIONS),
        auth_reachable_without_verify=reachable_without("AUTH_DELIVERY", "VERIFY"), surge=surge_pressure(t_close=None),
        surge_5s=surge_pressure(t_close=5.0), surge_2s=surge_pressure(t_close=2.0))
    # --- validation (crude vs closed form)
    rng = np.random.default_rng(seed + 4); V = {}
    for k in ("S0_baseline_admin", "S1_interlock_basic", "S3_interlock_full"):
        Pv = PRESETS[k].copy(lam0=0.05, m_adm0=0.2, p_load=1e-2, p_assoc=0.02, lam_DU=5e-5, p_sw=1e-3)
        an = float(event_hazard(Pv, 0.3, 0.4, False)); n = 4_000_000; cr = crude_event_mc(Pv, 0.3, 0.4, False, n, rng); se = math.sqrt(an * (1 - an) / n)
        V[k] = dict(analytic=an, crude=cr, z=(cr - an) / se)
    R["validation"] = V
    # --- economics
    eco = {}
    for N in (5, 60):
        for kind in ("S1", "S3"):
            for lvl in ("low", "nom", "high"):
                eco[f"{kind}|N{N}|{lvl}"] = annual_cost(kind, N, lvl)
    R["annual_cost"] = eco
    rows = []
    for E in (3650, 7300, 14600):
        for f in (0.3, 1.0, 3.0):
            r = dict(E=E, f=f)
            for kind in ("S1", "S3"):
                c, dh = breakeven_event_cost(kind, 60, E, f); r[kind] = dict(breakeven=c, dh=dh)
            rows.append(r)
    R["econ_table"] = rows; fig_econ(fig("fig13_econ.png"))
    R["econ_cost_sens"] = {f"{kind}|{lvl}": breakeven_event_cost(kind, 60, 7300, 1.0, lvl)[0] for kind in ("S1", "S3") for lvl in ("low", "nom", "high")}
    json.dump(R, open(os.path.join(outdir, "results.json"), "w"), indent=2, default=float); return R


def demo():
    import matplotlib; matplotlib.use("Agg")
    P = PRESETS["S1_interlock_basic"]
    print("nominal hazard per 1e6:", {k: round(1e6 * nominal_hazard(p), 3) for k, p in PRESETS.items()})
    mf = MeanField(P); print("bistable window:", mf.bistable_window(), "| rho_nom:", round(mf.rho(), 2))
    e = run_ensemble(P, 100, 1); print("S1 pmfe median:", round(float(np.median(e["pmfe"])), 3))
    assert not permissive(False, True, True, True, True) and permissive(True, True, True, True, True)
    print("DEMO OK")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--demo", action="store_true")
    ap.add_argument("--reproduce", action="store_true"); ap.add_argument("--out", default="misfuel_out")
    ap.add_argument("--seed", type=int, default=7); a = ap.parse_args()
    if a.demo: demo()
    elif a.reproduce: reproduce2(a.out, a.seed)
    else: ap.print_help()
