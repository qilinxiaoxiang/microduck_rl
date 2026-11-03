# Included inputs and evidence limits

All startup inputs are inside `starter/`; no source step relies on an author-only absolute path.

- **A interaction:** fabricated labelled transcripts; executable parser; exact acquisition, wake and ASR source snapshots with hashes. Private wake-model weights and personal recordings are excluded. Optional future local speech models are public downloads; their availability/runtime compatibility must be checked in that increment. Transcripts do not prove speech recognition or person identity.
- **B vision:** unchanged detector, composite orange-part frame, body-mask negative, generated non-ball challenge, protocol tracks; gaze/approach/tracking reference source and seven original fake-robot tests. No real ball/person benchmark. Historical camera settings are reference snapshots, not authority for the changed robot.
- **C/D motion:** four actual inference exports (walk61, t2b-best, stand61, spx-b_1800); 1,956 numeric rows extracted from bounded historical local traces; exact environment/worker/runtime snapshots; synthetic inference inputs and expected outputs. Complete hashes are in `motion/PROVENANCE.json`. No raw61 observation log or optimizer state; reconstructed inference is approximate. No missing dock-e/f checkpoint is claimed recovered.
- **E composition:** actual parser, saved detector readback, coordinator and labelled physical event fixtures; existing runtime source snapshots for the next adapter. The `within_entry`, supported exit and take signals in fixtures are not live estimators.

Historical source locations remain in provenance as audit references; they are not required for starter execution. Current physical facts belong only to `microduck-sim/CURRENT-STATE.json` in the owning robotics workspace. The packaged snapshot is dated 2026-11-03. New scan/masses, matched body, healthy runtime measurements and live sensor/readback validation are not supplied or claimed complete.

Starter dependencies are pinned in `starter/requirements.txt`, with component-level provenance under each folder. `starter/check_all.py` is the single combined acceptance entrypoint for recipient startup. Review evidence is in `ACCEPTANCE.md`; physical capability gates remain in the task cards and `BODY-READINESS.md`.
