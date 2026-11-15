# nq_sdpo-qwen2.5-7b-em-replay

Pure SDPO (Search-R1) with **Qwen/Qwen2.5-7B**.

| Artifact | Path |
|----------|------|
| Checkpoints | `verl_checkpoints/nq_sdpo-qwen2.5-7b-em-replay/` |
| Train replays | `res/train_replays/nq_sdpo-qwen2.5-7b-em-replay/` (`step_NNNN.jsonl`) |
| Train log | `exps/nq_sdpo-qwen2.5-7b-em-replay/train.log` |
| Experiment info | `exps/nq_sdpo-qwen2.5-7b-em-replay/info.json` |
| Launcher | `scripts/experiments/rui_meng/train_sdpo_v02_7b_replay.sh` |

Re-run:

```bash
cd /root/Search-R1
bash scripts/experiments/rui_meng/train_sdpo_v02_7b_replay.sh
```
