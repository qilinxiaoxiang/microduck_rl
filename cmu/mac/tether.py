"""Charger-lead tether on the MicroDuck trunk: one model for the eval sim (numpy) and the fine-tune (torch).

The robot runs on a USB-C lead (braided, about 3 m) from a power bank standing on the floor, plugged into the top of
the bank (about 0.18 m up); the lead leaves the bench backpack at its lower rear corner (Shawn 2026-10-30: the left
side; D-007 put the 5991 socket in the -y wall, so both sides are sampled) and lies on the floor between them.
In-place turns on the robot 2026-10-30 stalled after 40-90 deg (`turn-inplace-20261030/README.md`); the as-built sim
reproduces that only with a restoring yaw torque of about 0.010 N m/rad, a lead twisted by the turn.

Per step the lead puts on the trunk, at the exit point p (world):
  hanging   F_z = -lam g h_exit                       weight of the drop from the exit to the floor
  drag      F_xy = -mu lam g L_drag v_xy / (|v_xy| + V0)   floor friction on the length the exit drags along;
            L_drag = drag_len + the exit's distance from where it started, up to the lead on the floor: a short walk
            moves only the near part of the lead, a long one drags all of it (Shawn 2026-10-30); a snag (kink,
            the lead caught under the bank) multiplies the drag by snag_gain for a few seconds at random
  taut      F_xy += K_TAUT (d - L_free) toward the bank, when the exit is farther than the lead reaches, at most
            bank_slide (the bank then slides after the robot)
  twist     tau_z = -k_tw (yaw - yaw_ref), at most tau_slip; past it the lead end slips and yaw_ref follows
and the torque about the trunk COM is (p - com) x F + tau_z. Parameters are drawn per episode (`sample`).
Works on numpy arrays or torch tensors of shape (N, ...); `xp` is numpy or torch.
"""
import math

G = 9.81
V0 = 0.02            # m/s: velocity below which the drag fades (stick), avoids chatter at rest
K_TAUT = 50.0        # N/m once the lead is taut between bank and exit
LEAD_M = 3.0         # m
BANK_TOP_M = 0.18    # m: the plug at the top of the power bank
EXIT_B = (-0.060, 0.015, -0.030)  # trunk_base frame, m: backpack lower rear corner, left (+y); y sign sampled

RANGES = dict(
    lam=(0.02, 0.05),          # kg/m, braided USB-C leads 60-150 g over 3 m (not weighed)
    mu=(0.3, 0.8),             # lead on floor
    drag_len=(0.1, 0.6),       # m of lead dragged with the exit before it has moved
    snag_gain=(2.0, 4.0),      # drag multiplier while snagged
    snag_rate=0.05,            # snags per second
    snag_s=(1.0, 4.0),         # snag duration
    k_twist=(0.0, 0.02),       # N m/rad; the robot's stall fits 0.010
    tau_slip=(0.005, 0.03),    # N m before the lead end slips round; a lead that never slips is not trained for (Shawn 2026-10-30: too extreme)
    bank_dist=(0.3, 2.5),      # m from the robot's start, any bearing
    bank_slide=(0.5, 1.5),     # N to slide the bank on the floor (0.15-0.3 kg bank, mu 0.3-0.5)
    exit_jitter=0.010,         # m, each axis
)


def _clamp(xp, x, lo, hi):
    return x.clamp(lo, hi) if hasattr(x, "clamp") else xp.clip(x, lo, hi)


def sample(xp, n, rng_uniform, ranges=RANGES):
    """Per-episode parameters. rng_uniform(lo, hi, shape) -> array; returns a dict of (n,) or (n, 3) arrays."""
    u = lambda key: rng_uniform(ranges[key][0], ranges[key][1], (n,))
    side = xp.where(rng_uniform(0.0, 1.0, (n,)) < 0.5, -1.0 + 0 * u("mu"), 1.0 + 0 * u("mu"))
    j = ranges["exit_jitter"]
    exit_b = xp.stack([EXIT_B[0] + rng_uniform(-j, j, (n,)), side * EXIT_B[1] + rng_uniform(-j, j, (n,)),
                       EXIT_B[2] + rng_uniform(-j, j, (n,))], -1)
    bearing = rng_uniform(-math.pi, math.pi, (n,))
    return dict(lam=u("lam"), mu=u("mu"), drag_len=u("drag_len"), k_twist=u("k_twist"), tau_slip=u("tau_slip"),
                bank_dist=u("bank_dist"), bank_bearing=bearing, exit_b=exit_b, snag_gain=u("snag_gain"), bank_slide=u("bank_slide"))


def start(xp, prm, root_xy, yaw, exit_xy):
    """Episode start: place the bank relative to the robot and zero the twist. Returns the state dict."""
    bank = xp.stack([root_xy[:, 0] + prm["bank_dist"] * xp.cos(prm["bank_bearing"]),
                     root_xy[:, 1] + prm["bank_dist"] * xp.sin(prm["bank_bearing"])], -1)
    return dict(bank_xy=bank, yaw_ref=yaw * 1.0, exit0_xy=exit_xy * 1.0, snag_left=yaw * 0.0)


def step_snags(xp, state, dt, rng_uniform, ranges=RANGES):
    """Advance the snag timers by dt: a snag starts with probability snag_rate dt and lasts snag_s."""
    n = state["snag_left"].shape[0]
    new = (rng_uniform(0.0, 1.0, (n,)) < ranges["snag_rate"] * dt) & (state["snag_left"] <= 0)
    dur = rng_uniform(ranges["snag_s"][0], ranges["snag_s"][1], (n,))
    state["snag_left"] = xp.where(new, dur, _clamp(xp, state["snag_left"] - dt, 0.0, 1e9))


def wrench(xp, prm, state, p_exit, v_exit, com, yaw):
    """World-frame force (N, 3) and torque about the COM (N, 3); updates state["yaw_ref"] (slip)."""
    h = _clamp(xp, p_exit[:, 2], 0.0, 1.0)
    w = prm["lam"] * G
    vxy = v_exit[:, :2]
    speed = xp.sqrt((vxy ** 2).sum(-1))
    free = LEAD_M - BANK_TOP_M - h
    moved = xp.sqrt(((p_exit[:, :2] - state["exit0_xy"]) ** 2).sum(-1))
    length = _clamp(xp, prm["drag_len"] + moved, 0.0, LEAD_M - BANK_TOP_M)
    gain = xp.where(state["snag_left"] > 0, prm["snag_gain"], 1.0 + 0 * prm["snag_gain"])
    drag = (-prm["mu"] * w * length * gain / (speed + V0))[:, None] * vxy
    to_bank = state["bank_xy"] - p_exit[:, :2]
    d = xp.sqrt((to_bank ** 2).sum(-1)) + 1e-6
    tension = xp.minimum(K_TAUT * _clamp(xp, d - free, 0.0, 1e3), prm["bank_slide"])
    pull = (tension / d)[:, None] * to_bank
    # the bank slides so the lead never stretches past its length
    over = _clamp(xp, d - free - prm["bank_slide"] / K_TAUT, 0.0, 1e3)
    state["bank_xy"] = state["bank_xy"] - (over / d)[:, None] * to_bank
    fxy = drag + pull
    f = xp.stack([fxy[:, 0], fxy[:, 1], -w * h], -1)
    # twist with slip: the lead end turns round once the twist torque exceeds tau_slip
    k = prm["k_twist"]
    lim = prm["tau_slip"]
    tw = _clamp(xp, -k * (yaw - state["yaw_ref"]), -lim, lim)
    slipping = xp.abs(k * (yaw - state["yaw_ref"])) > lim
    state["yaw_ref"] = xp.where(slipping, yaw + tw / (k + 1e-9), state["yaw_ref"])
    r = p_exit - com
    tq = xp.stack([r[:, 1] * f[:, 2] - r[:, 2] * f[:, 1],
                   r[:, 2] * f[:, 0] - r[:, 0] * f[:, 2],
                   r[:, 0] * f[:, 1] - r[:, 1] * f[:, 0] + tw], -1)
    return f, tq
