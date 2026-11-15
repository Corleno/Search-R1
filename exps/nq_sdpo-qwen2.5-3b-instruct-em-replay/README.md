# nq_sdpo-qwen2.5-3b-instruct-em-replay

Pure SDPO (Search-R1) with **Qwen/Qwen2.5-3B-Instruct**.

| Artifact | Path |
|----------|------|
| Checkpoints | `verl_checkpoints/nq_sdpo-qwen2.5-3b-instruct-em-replay/` |
| Train replays | `res/train_replays/nq_sdpo-qwen2.5-3b-instruct-em-replay/` (`step_NNNN.jsonl`) |
| Train log | `exps/nq_sdpo-qwen2.5-3b-instruct-em-replay/train.log` |
| Experiment info | `exps/nq_sdpo-qwen2.5-3b-instruct-em-replay/info.json` |
| Launcher | `scripts/experiments/rui_meng/train_sdpo_v02_instruct_replay.sh` |

Re-run:

```bash
cd /root/Search-R1
bash scripts/experiments/rui_meng/train_sdpo_v02_instruct_replay.sh
```
