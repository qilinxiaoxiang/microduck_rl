# Mochi response: deployment verified 2026-11-04

Mochi name response is deployed on the team's A7Z robot. The service is enabled for boot, and the owner confirmed audible quacks and head turns in the live check. This advances P5/P6 by removing manual listener startup; it is a component milestone in the fetch project.

## Behavior and evidence

- After boot audio selfcheck, the service listens and quacks in audio-only mode when no ready motor brain exists. It never starts the motor brain or makes the robot stand up.
- A fresh zero-speed walking-policy state or a fully seated state enables guarded head responses. Other states use audio-only listening. This does not implement stopping a walking robot on its name.
- The supervisor releases its capture process before changing modes and waits while the audio selfcheck owns the device. The deployed spotter is `mochi_v4.onnx`, selected explicitly with `--spotter oww`; this differs from the generic program's `own` default.
- Live evidence: three detected calls, three unmuted quack events and face detections. The owner confirmed the audible and visible response. This small check does not measure a general detection rate or false-alarm rate.
- Service readback was `enabled` and `active`; a service restart returned to one listener and one microphone capture process. Installed source hashes matched. The 77 name-response and supervisor regression checks passed.

Actual cold power-on and seated physical response were not exercised in this session. Boot configuration is verified; cold-boot physical acceptance remains open. Extreme-angle gaze, noisy walking calls, caller identity, spoken fetch requests, loaded return and handover are not accepted by this check.

## Implication for Task A

Reuse the existing name-response capability. Task A still needs one timestamped audio source shared between wake detection and speech transcription, resolved caller identity and versioned events for the coordinator. Face detection alone is not caller identity. Do not open a second robot microphone capture beside the running service.

The [interaction source package](planning/fetch-20261103/starter/interaction/reference/README.md) remains the exact **2026-11-03 offline reference snapshot**, with its original nine-file hash manifest. It is not the deployed service bundle. Its CPU-only parser and source verification still run without private weights, recordings or robot access. This status update does not add native-service installation support to the RL repository.

## Provenance and operator readback

Canonical runtime source is maintained in the robotics workspace, implementation commit `1c3bfef8`:

- `microduck-sim/guide/firmware/mac/namecall.py`
- `microduck-sim/guide/firmware/mac/namecall_boot.py`
- `microduck-sim/guide/firmware/pi/systemd/duck-namecall.service`
- `microduck-sim/guide/firmware/mac/namecall-20261029/install_namecall.sh`

The owner maintains current configuration in `microduck-sim/CURRENT-STATE.json`, fact `a7z_namecall_start`, and acceptance in `namecall-20261029/REQUIREMENTS.json` R16 plus `boot-acceptance-20261104.json`. Those paths identify the canonical records; teammate starter checks do not require access to them. R16 is still partial because a cold-boot observation is outstanding.

On an already authorized robot shell, read service status without changing motor state:

```sh
systemctl show duck-namecall -p ActiveState -p UnitFileState -p NRestarts
journalctl -u duck-namecall -n 20 --no-pager
```

Deployed SHA-256 values:

- `namecall.py`: `491724cd5d565e514adb13048152cf62f8fb3940445537743e0b9f730f2168a0`
- `namecall_boot.py`: `0c104109a38a073683fa66ccb46a391f5a22d36bde8f394c0c43487bcd6fa3d4`
- `duck-namecall.service`: `7a278a88e505e40da8ba6143194893862221b6e85523abedf64f958f43b55556`
