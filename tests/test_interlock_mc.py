import math, itertools, numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from interlock_mc import *

def test_failsafe_truth_table():
    # relay energises only when every affirmative condition holds; any single False de-energises
    base = dict(tag_valid=True, record_found=True, product_match=True, operator_auth=True, heartbeat_ok=True)
    assert permissive(**base)
    for k in base: assert not permissive(**{**base, k: False}), k
    # override works only with BOTH token and 2FA, and only when tag invalid
    nb = {**base, "tag_valid": False}
    assert not permissive(**nb) and not permissive(**nb, override_token=True)
    assert permissive(**nb, override_token=True, supervisor_2fa=True)
    assert not permissive(**{**nb, "heartbeat_ok": False}, override_token=True, supervisor_2fa=True)

def test_closed_form_matches_crude_mc():
    rng = np.random.default_rng(1)
    for k in ("S0_baseline_admin", "S1_interlock_basic", "S3_interlock_full"):
        P = PRESETS[k].copy(lam0=0.05, m_adm0=0.2, p_load=1e-2, p_assoc=0.02, lam_DU=5e-5, p_sw=1e-3)
        an = float(event_hazard(P, 0.3, 0.4, False)); n = 3_000_000
        cr = crude_event_mc(P, 0.3, 0.4, False, n, rng); se = math.sqrt(an * (1 - an) / n)
        assert abs(cr - an) < 4 * se, (k, an, cr)

def test_barrier_ordering_and_compromise_floor():
    h = {k: nominal_hazard(P) for k, P in PRESETS.items()}
    assert h["S0_baseline_admin"] > h["S1_interlock_basic"] > h["S3_interlock_full"]
    P = PRESETS["S1_interlock_basic"]
    # a compromised interlock must never be worse than administrative checks alone
    assert float(event_hazard(P, .3, .4, True)) <= float(event_hazard(PRESETS["S0_baseline_admin"], .3, .4, False)) + 1e-15

def test_meanfield_window_and_linear_limit():
    mf = MeanField(Params())
    w = mf.bistable_window(); assert w and 3.8 < w[0] < 4.0 and 11.2 < w[1] < 11.4   # inherits MyUncle defaults
    lin = MeanField(Params(beta_c=0.0))
    for rho in (0.5, 2.0, 9.0):
        fp = lin.fixed_points(rho); assert len(fp) == 1 and abs(fp[0] - 1 / (1 + rho)) < 1e-3
    assert len(mf.branches(7.0)) == 2 and len(mf.branches(20.0)) == 1 and len(mf.branches(1.0)) == 1

def test_more_auditing_reduces_compromise():
    P = PRESETS["S1_interlock_basic"]
    a = run_ensemble(P.copy(T_detect=90.0), 60, 3)["comp_mean"].mean()
    b = run_ensemble(P.copy(T_detect=7.0), 60, 3)["comp_mean"].mean()
    assert b < a

def test_drills_rule():
    assert drills_required(1e-2) == 299 and drills_required(1e-3) == 2995


# ---- v2 tests ----
def test_state_machine_properties():
    assert not reachable_without("AUTH_DELIVERY", "VERIFY")            # start only via verification
    sm = StateMachine()
    for ev in ("tag_scanned", "verify_ok"): sm.fire(ev)
    assert sm.state == "AUTH_DELIVERY" and StateMachine.start_permit_energised(sm.state)
    sm.fire("controller_fault"); assert sm.state == "FAULT" and not StateMachine.start_permit_energised(sm.state)
    sm.fire("tag_scanned"); assert sm.state == "FAULT"                  # undefined event ignored; no new start from FAULT
    for ev in ("supervisor_reset", "self_test_ok"): sm.fire(ev)
    assert sm.state == "IDLE"
    sm = StateMachine()
    for ev in ("tag_scanned", "verify_ok", "product_veto", "flow_stopped"): sm.fire(ev)
    assert sm.state == "RECOVERY"
    only_auth = [s for s in STATES if StateMachine.start_permit_energised(s)]; assert only_auth == ["AUTH_DELIVERY"]

def test_surge_controlled_closure_reduces_pressure():
    a = surge_pressure(); b = surge_pressure(t_close=5.0)
    assert b["dP"] < 0.05 * a["dP"] and a["t_crit"] < 0.1

def test_fault_injection_structural_categories():
    r = fault_injection_campaign(4000, 1); full = r["full"]["categories"]
    for c in ("stale_record", "comms_loss", "override_attack", "power_interruption"): assert full[c]["unsafe"] == 0
    assert r["none (all defences off)"]["p"] > 0.8 and r["full"]["p"] < 0.1
    assert r["full minus two_factor"]["categories"]["override_attack"]["p"] > 0.4

def test_dependence_conclusion_survives():
    d = dependence_grid()
    assert min(v["rrr"] for v in d["S1_interlock_basic"].values()) > 0.7 and min(v["rrr"] for v in d["S3_interlock_full"].values()) > 0.95

def test_paired_draws_and_batch_shapes():
    a = run_ensemble(PRESETS["S1_interlock_basic"], 40, 5, n_stoch=2); b = run_ensemble(PRESETS["S1_interlock_basic"], 40, 5, n_stoch=2)
    assert np.allclose(a["pmfe"], b["pmfe"]) and a["n_par"] == 20 and a["pmfe"].shape == (40,)
    lo, hi = wilson(0, 1000); assert lo == 0 and 0 < hi < 0.01

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
