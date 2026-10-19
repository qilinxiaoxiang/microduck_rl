# Kaggle launcher for the walk61 fine-tune

`kaggle_job.py` runs `cmu/finetune/modal/finetune_worker.py` on Kaggle's free T4 GPUs with the same flags as
`cmu/finetune/modal/finetune.py`. Usage is in its docstring; the general Kaggle procedure is the `kaggle-gpu-jobs` skill.
Run it from the repository root.

How it works: `pack` uploads the repository (without venvs, logs, runs, checkpoints, `cmu/real-logs`, the
`cmu/whole/runtime-overlay/models-ft*` candidates and `*.gz`) plus the four assets from `cmu/whole/runtime-overlay/models`,
`cmu/whole/dynamics/asbuilt-20261015` and `cmu/whole/electrical`, and `cmu/model/manifests/*.json` when present (the
`--cmu-config plugged|battery` manifests, mounted as `/root/assets/manifests/`), as the private dataset
`shawnxiang9527/microduck-rl-src`, skipped when the content hash is unchanged. `pack --dry-run` stages the tarball and
lists it without the Kaggle CLI or a token. `push` writes a private script
kernel `microduck-<run-name>` around `kernel_entry.py`, which runs `uv sync --frozen --no-dev` (Python 3.12 venv, about
90 s), an import check, then the worker with `--root /kaggle/working/runs --rate-per-hour 0`. The dollar cap is
therefore inactive on Kaggle; `--max-hours` is the cap and must stay at or below 11.5 (12 h session limit).
`fetch` downloads `/kaggle/working` to `cmu/finetune/kaggle/out/<run-name>/`; `job_record.json` names the failed stage.

Paired jobs: `push <preset> --run-name A --pair-run-name B -- <A flags> -- <B flags>` runs both workers in one
session, A on GPU 0 and B on GPU 1 (`CUDA_VISIBLE_DEVICES`), so one quota hour trains two jobs. Both use the same
preset; the launcher refuses a pair with identical flags. The kernel is named after A, so `status A` and `fetch A`
cover both. Output: `runs/A/`, `runs/B/`, `train_A.log`, `train_B.log`, `gpu_usage.csv` (both GPUs every 30 s) and
`job_record.json["jobs"]` with each worker's exit code; the kernel ends ERROR when either job fails, and stops with
`gpu_count` when the session has fewer than two GPUs.

## Measured 2026-10-18

- smoke-1018b (64 envs, 6 iterations): COMPLETE in 4.3 min end to end, checkpoints and ONNX exported.
- bench-t4-1018 (4096 envs, 40 iterations): 0.211 it/s (4.7 s per iteration) on one Tesla T4. Modal L4 on the same
  worker: 0.31-0.32 it/s (ft3, FINETUNE-DELTA.md). 1000 iterations take about 80 min on T4.
- `--accelerator NvidiaL4` gave a session without a GPU (no nvidia-smi) on this free account; use the default
  `NvidiaTeslaT4`. The session has two T4s; the worker uses one.
- Three sessions (about 15 min) used 0.26 h of the 30 h weekly GPU quota.
- smoke-1018c (64 envs, `--delta lowcmd --max-lr 0.0003 --track-guard 1.3`): COMPLETE, the kernel's source digest
  matched the uploaded dataset, ONNX exported. `kaggle quota` read 0.35 h used after this run (five sessions in total).
- smoke-1018-0108 found a worker bug, fixed in `cmu/finetune/modal/finetune_worker.py`: with `--track-guard 0` (guard off, the
  default) the save hook divided by zero once the low-command baseline existed. The fix skips that check when the
  guard is off; Modal runs with the guard on (ft3, 1.3) were not affected.
- psmoke-a / psmoke-b (paired smoke, seeds 1 and 2): COMPLETE in 4 min, ONNX in both run dirs, `gpu_usage.csv` shows
  memory and load on GPU 0 and GPU 1, quota 0.35 h to 0.42 h (one session).
- pbench-a / pbench-b (paired bench, 4096 envs, 40 iterations): 0.212 and 0.206 it/s, against 0.211 for one job alone;
  both GPUs averaged 54 % utilisation. Session 9 min, quota 0.42 h to 0.57 h. Two jobs per session cost the same
  quota as one.
