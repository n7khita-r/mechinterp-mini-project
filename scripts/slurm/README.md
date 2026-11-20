# Running on Ada (IIIT-H)

Short version:

```bash
# login node, once
bash scripts/slurm/setup_env.sh
export HF_TOKEN=hf_xxxxxxxx
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-3-4b-pt')"

# then
sbatch scripts/slurm/00_env_check.sbatch        # READ THE OUTPUT before continuing
sbatch scripts/slurm/run_all_4b.sbatch          # 3-5 h on one 2080 Ti
```

Four GPUs free instead of one:

```bash
JID=$(sbatch --parsable scripts/slurm/run_split_4b.sbatch)
sbatch --dependency=afterok:$JID scripts/slurm/finalise_4b.sbatch
```

---

## What Ada forces on the design

| Constraint | Consequence |
|---|---|
| GPUs are GTX 1080 Ti (SM 6.1) and RTX 2080 Ti (SM 7.5) | **No bf16** — that needs Ampere/SM 80. Gemma 3 trained in bf16, so we run fp16 and *verify* it numerically in stage 00. |
| 1080 Ti fp16 runs at 1/64 rate | Every script passes `--exclude=gnode[01-42]`. Don't remove it. |
| 11 GB per card | 4B fp16 with the vision tower dropped is ~7.8 GB, leaving ~3 GB. `head_batch: 8`, `score: 8`, and `logits_to_keep=1` (Gemma's 262k vocab makes full logits cost 1.7 GB per batch-8 forward). |
| 4 GPUs max per job (QoS medium/normal) | 27B (~54 GB) cannot run here at all. |
| `/home` quota 50 GB | 4B weights ~8.6 GB: fine. 27B ~54 GB: does not fit. |
| `/share1` is master-node only | `HF_HOME` **must** be under `/home`, or compute nodes can't see the weights. |
| `/scratch`, `/ssd_scratch` purged after 7 days | Never put weights or results there. |
| `srun` capped at 6 h | The full run must go through `sbatch`. |
| QoS `medium`: 4 GPUs, 4-day walltime | Comfortable. Check yours with `sacctmgr show assoc user=$USER format=Account,QOS,DefaultQOS`. |
| Ubuntu 18.04 → glibc 2.27 | **torch 2.3.1 is the ceiling.** 2.4+ wheels are manylinux_2_28 and will not import. Pinned in `constraints-ada.txt`. |

## Files here

| File | Purpose |
|---|---|
| `setup_env.sh` | One-time venv on the login node, with the Ada version pins |
| `constraints-ada.txt` | pip constraints that keep torch at 2.3.1 |
| `00_env_check.sbatch` | Sanity + the fp16-vs-fp32 verification. Run first. |
| `run_all_4b.sbatch` | All stages, one GPU, 3-5 h |
| `run_split_4b.sbatch` | Array job, one condition per GPU, 1-2 h |
| `finalise_4b.sbatch` | Pooled AIE + transfer + report; depends on the array |
| `../run_27b.sh` | 27B elsewhere (H100 80 GB), plain bash |

## Reading stage 00's output

Three lines decide whether to continue:

```
[PASS] not a Pascal GPU                   compute capability 7.5
[PASS] fp16 produces no NaN/Inf           checked English, Telugu-script and romanized probes
[PASS] fp16 agrees with fp32 reference    top-1 match=True, min logit corr=0.99998
[INFO] peak 8.9 GiB at batch 8; head_batch up to ~11 should fit
```

- **If the fp16 agreement check FAILS**, stop. Gemma 3's activations are
  overflowing fp16 and every number downstream would be suspect. Move to a
  bf16-capable GPU and use `configs/model/gemma3_4b_bf16.yaml`.
- **If "not a Pascal GPU" fails**, you landed on a 1080 Ti. Resubmit; the
  `--exclude` should prevent it.
- The `head_batch` recommendation is measured, not guessed. If it comes back
  below 8, lower `batch.head_batch` in the model config before the long run.

## If a job dies

Every stage caches results keyed by `(item, config hash)`, so resubmitting the
same script resumes rather than restarting. Nothing needs cleaning up first.

To watch a running job:

```bash
squeue -u $USER
tail -f slurm_logs/fvcm-4b-<jobid>.out
```

Common pending reasons: `Priority` (waiting your turn), `Resources` (no free
GPU), `QOSMaxJobsLimit` (you already have 4 jobs running under QoS medium).

## Getting results back

```bash
rsync -avz nikhitaravi@ada-gw1.iiit.ac.in:MechInterp-final-project/results/ results/
rsync -avz nikhitaravi@ada-gw1.iiit.ac.in:MechInterp-final-project/paper/figures/ paper/figures/
rsync -avz nikhitaravi@ada-gw1.iiit.ac.in:MechInterp-final-project/paper/tables/ paper/tables/
```

Then `make paper` and `make slides` locally.

---

## Cluster facts, as actually observed (2026-11-20)

The Ada wiki is out of date in several ways that each broke a job submission.
What is true in practice:

| Wiki says | Reality |
|---|---|
| partitions `short` / `long` | `u22` (default, **infinite** timelimit), `u22-cpu`, `ihub`, `plafnet2`, `rrc` |
| modules `u18/python/3.11.2`, `u18/cuda/11.6` | no `u18/*` modules; the cluster is on Ubuntu 22.04 (hence `u22`). `setup_env.sh` probes for what exists and falls back to the system Python. |
| node names `gnode01`-`gnode92` | **three digits**: `gnode001`-`gnode118` |
| `/home` quota 25 or 50 GB | home is `/home2/$USER`; 50 GB |
| nodes 01-40 are 1080 Ti, 43-92 are 2080 Ti | **unverifiable.** gres is just `gpu:N` with no model, and `AVAIL_FEATURES` is `(null)`, so neither `--constraint` nor a reliable `--exclude` is possible. |

Because the GPU model cannot be requested, no node exclusion is set. Instead
`scripts/00_env_check.py` asserts compute capability >= 7 and prints the card
name, so landing on a Pascal card fails in under a minute -- resubmit, and note
the node number in the table below as the real mapping is learned.

Observed node -> GPU (fill in as you go):

| node | GPU | compute capability |
|---|---|---|
|  |  |  |

Also: GPU counts per node vary (2, 3 or 4), so `run_split_4b.sbatch`, which needs
four separate single-GPU allocations, queues longer than `run_all_4b.sbatch`.
