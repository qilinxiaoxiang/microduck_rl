"""Round 3 (2026-11-19): the terms that close the action-chatter gap of the flight policies.

s2r6_4 stood and walked in every simulated cell and fell on the robot at standstill in 4.4 s with 172 action jumps
over 0.5 rad per 10 s; the runtime simulation shows the same fall only with the servo position gain doubled
(extensions/flight-20261110/train/RUNLOG.md "Round 3", CHATTER-LITERATURE.md). Three pieces, each a delta key of
flight_worker.apply_delta:

  gain_dr {"kp": [lo, hi]}          one position-gain scale per env per episode, the same on every servo
                                    (the base recipe's event averages 14 per-joint draws, which collapses a wide
                                    range to its middle; its switch is off)
  reward_mods.action_acc {"weight"} second difference of the actions, squared and summed (Disney BDX reward table:
                                    action rate and action acceleration together)
  anchor61 {"onnx", "beta", ...}    behaviour-cloning anchor to the deployed walk61 on no-pack states, one supervised
                                    step after every PPO update (Wolczyk et al. ICML 2024; the t2 anchor of
                                    2026-10-30, which brought the action distance on walk61's robot walks to 0.208 rad)
"""
from __future__ import annotations

import torch

ACTOR_HEAD = slice(51, 55)  # head-pose command columns of the 61 actor inputs
LEG_OUT = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]  # action outputs of the two legs (5-8 are neck and head)
ANCHOR = {}  # the anchor61 dict of the delta being trained, set by flight_worker.apply_delta


def action_acc_l2(env):
    """sum_j (a_t - 2 a_t-1 + a_t-2)_j^2 (a cost: >= 0, used with a negative weight)"""
    am = env.action_manager
    return torch.sum(torch.square(am.action - 2.0 * am.prev_action + am.prev_prev_action), dim=1)


def randomize_kp_scale(env, env_ids, kp_range):
    """reset event: one position-gain scale per env, the same on every BAM servo of the robot (non-accumulating:
    reset_gains first, as mdp.randomize_delayed_actuator_gains). The velocity gain keeps its nominal value: the
    matched runtime cell (--kp-rl 400) changes the position gain only."""
    from bam.mjlab import BamActuator
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)
    lo, hi = float(kp_range[0]), float(kp_range[1])
    s = torch.rand(len(env_ids), 1, device=env.device) * (hi - lo) + lo
    n = 0
    for act in env.scene["robot"].actuators:
        if isinstance(act, BamActuator):
            act.reset_gains(env_ids)
            act.set_gains(env_ids, kp_scale=s, kd_scale=torch.ones_like(s))
            n += 1
    assert n > 0, "gain_dr: the robot has no BAM actuator to scale"
    if not hasattr(env, "_kp_scale"):
        env._kp_scale = torch.ones(env.num_envs, 1, device=env.device)
    env._kp_scale[env_ids.long()] = s


def actor_mean(actor, x):
    """deterministic action of an rsl_rl MLPModel actor on raw concatenated actor inputs"""
    return actor.mlp(actor.obs_normalizer(x))


def anchor_batch(x, ref, cfg, payload_kg=None):
    """Rows, targets and per-output weights of one anchor step.

    x: raw actor inputs [M, 61 or 61 + 7] of a rollout. Kept rows: payload at most cfg["payload_max_kg"] (the teacher's
    own privileged column 61, or payload_kg [M] for a 61-input actor). Target: walk61 on the same 61 inputs with the
    head command zeroed (walk61 does not step while a head command is set, FINETUNE-DELTA.md "Run t3 plan"). Weights:
    beta on the ten leg outputs of every kept row, beta on the four head outputs only where the head command is
    exactly zero (elsewhere the head follows its command)."""
    kg = x[:, 61] if x.shape[1] > 61 else payload_kg
    keep = kg <= float(cfg.get("payload_max_kg", 0.02))
    x = x[keep]
    if len(x) > int(cfg.get("batch", 4096)):
        x = x[torch.randperm(len(x), device=x.device)[: int(cfg.get("batch", 4096))]]
    x0 = x[:, :61].clone()
    head_default = (x0[:, ACTOR_HEAD] == 0).all(dim=1)
    x0[:, ACTOR_HEAD] = 0.0
    with torch.no_grad():
        mu0 = ref(x0)
    w = torch.zeros(len(x), 14, device=x.device, dtype=x.dtype)
    w[:, LEG_OUT] = float(cfg["beta"])
    w[head_default] = float(cfg["beta"])
    return x, mu0, w


def patch_runner_for_anchor():
    """The registry's runner class with the anchor step after every PPO update (not during the critic warm-up, while
    the worker's runner has the actor frozen). Idempotent; wraps whatever load_runner_cls returns at call time, so it
    composes with flight_worker's history patch."""
    import mjlab.tasks.registry as reg
    if getattr(reg, "_anchor_runner_patched", False):
        return
    orig = reg.load_runner_cls

    def load_runner_cls(task):
        from mjlab.rl import MjlabOnPolicyRunner
        base = orig(task) or MjlabOnPolicyRunner
        if not ANCHOR:
            return base

        class AnchorRunner(base):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                import teacher as tc
                cfg = dict(ANCHOR)
                self._anchor_cfg = cfg
                self._anchor_ref = tc.TorchPolicy(cfg["onnx"]).to(self.device)
                assert self._anchor_ref.width == 61, self._anchor_ref.width
                self._anchor_sum, self._anchor_n, self._anchor_rows, self._anchor_updates = 0.0, 0, 0, 0
                orig_update = self.alg.update

                def update():
                    batch = None
                    if not getattr(self, "warm", False):
                        batch = self._anchor_collect()  # before update(): it clears the rollout storage
                    out = orig_update()
                    if batch is not None:
                        loss = self._anchor_step(*batch)
                        if isinstance(out, dict):
                            out["anchor"] = loss
                    return out
                self.alg.update = update
                print(f"[anchor] walk61 anchor on rows with payload <= {cfg.get('payload_max_kg', 0.02)} kg, beta "
                      f"{cfg['beta']}, batch {cfg.get('batch', 4096)}, target {cfg['onnx']}", flush=True)

            def _anchor_collect(self):
                actor, obs = self.alg.actor, self.alg.storage.observations
                x = torch.cat([obs[g] for g in actor.obs_groups], dim=-1)
                kg = None
                if x.shape[-1] == 61:  # a one-frame actor: the env's payload draw, per env
                    kg = self.env.unwrapped._payload_obs[:, 0].detach().unsqueeze(0).expand(x.shape[0], -1).flatten()
                x, mu0, w = anchor_batch(x.flatten(0, 1).detach().clone(), self._anchor_ref, self._anchor_cfg, kg)
                return (x, mu0, w) if len(x) >= 64 else None

            def _anchor_step(self, x, mu0, w):
                actor = self.alg.actor
                loss = (w * torch.square(actor_mean(actor, x) - mu0)).sum(dim=1).mean()
                self.alg.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), self.alg.max_grad_norm)
                self.alg.optimizer.step()
                with torch.no_grad():  # leg-output distance to walk61 on the anchored rows, rad
                    d = float((actor_mean(actor, x) - mu0)[:, LEG_OUT].norm(dim=1).mean())
                self._anchor_sum += d
                self._anchor_n += 1
                self._anchor_rows += len(x)
                self._anchor_updates += 1
                return float(loss)

            def save(self, path, infos=None):
                super().save(path, infos)
                if self._anchor_n:
                    print(f"[anchor] it {self.current_learning_iteration} leg distance to walk61 on no-pack rows "
                          f"{self._anchor_sum / self._anchor_n:.3f} rad ({self._anchor_n} steps, "
                          f"{self._anchor_rows // self._anchor_n} rows per step)", flush=True)
                    self._anchor_sum, self._anchor_n, self._anchor_rows = 0.0, 0, 0

        return AnchorRunner

    reg.load_runner_cls = load_runner_cls
    reg._anchor_runner_patched = True
