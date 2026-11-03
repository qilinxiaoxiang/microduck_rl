"""Motor-free reference coordinator. Outputs intentions; never controls hardware.

The physical acknowledgements in the rehearsal are fixtures, not estimators.
A future runtime adapter must produce them from validated sensor readback.
"""
import math


class Fetch:
    def __init__(self, max_track_age=0.25):
        self.state = "idle"
        self.caller = None
        self.epoch = 1
        self.payload = False
        self.pending_fetch = False
        self.settled = False
        self.looked = False
        self.last_t = -1.0
        self.track = None
        self.max_track_age = max_track_age  # protocol proposal, not measured safe latency
        self.commands = []
        self.journal = []
        self.profile = "fixture-entry-v1"

    def emit(self, intent, **fields):
        self.commands.append(dict(v=1, epoch=self.epoch, t=self.last_t,
                                  intent=intent, **fields))

    def fresh(self, now):
        return bool(self.track and self.track["visible"] and
                    0 <= now - self.track["capture_t"] <= self.max_track_age)

    def target(self):
        return self.caller if self.payload else "orange-ball"

    def handle(self, event):
        before = self.state
        start = len(self.commands)
        t = event.get("t")
        if isinstance(t, bool) or not isinstance(t, (float, int)) or not math.isfinite(t) or t < self.last_t:
            raise ValueError("finite nondecreasing receive times required")
        self.last_t = t
        if event.get("v") != 1:
            raise ValueError("unsupported protocol version")
        if event.get("epoch") != self.epoch:
            self.journal.append(dict(t=t, event=event["kind"], before=before,
                                     after=self.state, outcome="revoked_epoch"))
            return []
        k = event["kind"]
        incoming_selected_track = (k == "track" and event.get("target_id") == self.target()
                                   and event.get("visible") and
                                   isinstance(event.get("capture_t"), (int, float)) and
                                   0 <= t - event["capture_t"] <= self.max_track_age)
        if not incoming_selected_track and self.state in ("approaching", "returning") and not self.fresh(t):
            self.state = "seeking_caller" if self.payload else "seeking_ball"
            self.emit("controlled_stop", reason="camera_timeout", keep_payload=self.payload)
        if k in ("manual_takeover", "emergency"):
            self.epoch += 1
            self.state = "stopped"
            self.emit("revoke_autonomy", reason=k)
        elif self.state in ("completed", "stopped", "failed"):
            pass  # explicit new session is required; stale events never restart a task
        elif k == "call":
            if self.state == "idle" and event.get("caller_id"):
                self.caller = event["caller_id"]
                self.state = "attending"
                self.emit("controlled_stop", preserve_posture=True)
            else:
                self.emit("busy", original_caller=self.caller)
        elif k == "request":
            if (event.get("caller_id") == self.caller and self.caller and
                    event.get("target_type") == "orange_ball" and
                    self.state in ("attending", "looking_at_caller", "ready")):
                self.pending_fetch = True
                if self.settled and self.looked:
                    self.state = "seeking_ball"
                    self.emit("find_target", target_id="orange-ball")
        elif k == "settled" and self.state == "attending":
            self.settled = True
            self.emit("gaze_only", target_id=self.caller, preserve_posture=True)
            self.state = "looking_at_caller"
        elif k == "caller_looked" and self.state == "looking_at_caller":
            ct = event.get("capture_t")
            if (event.get("caller_id") == self.caller and event.get("visible") and
                    event.get("gaze_aligned") and isinstance(ct, (int, float)) and
                    0 <= t - ct <= self.max_track_age):
                self.looked = True
                self.state = "seeking_ball" if self.pending_fetch else "ready"
                if self.pending_fetch:
                    self.emit("find_target", target_id="orange-ball")
        elif k == "cancel" and self.caller and event.get("caller_id") == self.caller:
            self.state = "cancel_pending"
            self.emit("request_supported_abort", keep_payload=self.payload)
        elif k == "abort_supported" and self.state == "cancel_pending":
            if event.get("supported") and event.get("settled"):
                self.state = "stopped"
                self.epoch += 1
                self.emit("revoke_autonomy", reason="cancel_complete")
        elif k == "track" and self.state in ("seeking_ball", "approaching", "seeking_caller", "returning"):
            if event.get("target_id") == self.target():
                ct = event.get("capture_t")
                if not isinstance(ct, (int, float)) or not math.isfinite(ct):
                    raise ValueError("capture_t required")
                self.track = event
                if self.fresh(t):
                    self.state = "returning" if self.payload else "approaching"
                    self.emit("follow_target", target_id=self.target(),
                              capture_t=ct, source=event.get("source"))
                else:
                    self.state = "seeking_caller" if self.payload else "seeking_ball"
                    self.emit("controlled_stop", reason="target_lost_or_stale", keep_payload=self.payload)
            else:
                self.emit("ignore_other_identity", target_id=event.get("target_id"))
        elif k == "tick" and self.state in ("approaching", "returning") and not self.fresh(t):
            self.state = "seeking_caller" if self.payload else "seeking_ball"
            self.emit("controlled_stop", reason="camera_timeout", keep_payload=self.payload)
        elif k == "dock_ready" and self.state == "approaching":
            if (self.fresh(t) and event.get("profile_id") == self.profile and
                    event.get("within_entry") and event.get("history_preserved") and
                    event.get("sample_id")):
                self.state = "picking"
                self.track = None
                self.emit("start_pick", entry_sample=event["sample_id"],
                          preserve_state=True, preserve_previous_action=True)
        elif k == "pick_result" and self.state == "picking":
            if all(event.get(x) for x in ("lifted", "held", "upright", "supported_exit")):
                self.payload = True
                self.state = "seeking_caller"
                self.emit("find_target", target_id=self.caller, keep_payload=True)
            elif event.get("supported_exit"):
                self.state = "seeking_ball"
                self.emit("controlled_stop", reason="empty_bite_reacquire")
            else:
                self.state = "cancel_pending"
                self.emit("request_supported_abort", reason="unsupported_pick_exit")
        elif k == "arrived" and self.state == "returning":
            if (self.fresh(t) and event.get("caller_id") == self.caller and
                    event.get("within_wait_zone") and event.get("settled") and self.payload):
                self.state = "waiting_take"
                self.emit("hold_for_human", caller_id=self.caller, jaw="keep_grip")
        elif k == "human_take" and self.state == "waiting_take":
            if (event.get("take_confirmed") and event.get("object_absent") and
                    event.get("caller_id") == self.caller):
                self.state = "completed"
                self.payload = False
                self.emit("delivery_complete", caller_id=self.caller)
        elif k == "payload_lost" and self.payload:
            self.state = "failed"
            self.payload = False
            self.emit("controlled_stop", reason="drop_not_delivery")
        elif k == "expression" and self.payload:
            self.emit("expression_without_jaw", reason="preserve_grip")
        self.journal.append(dict(t=t, event=k, before=before, after=self.state))
        return self.commands[start:]
