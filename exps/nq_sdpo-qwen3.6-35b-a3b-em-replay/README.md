# nq_sdpo-qwen3.6-35b-a3b-em-replay

Pure SDPO (Search-R1) with **Qwen/Qwen3.6-35B-A3B**.

| Artifact | Path |
|----------|------|
| Checkpoints | `verl_checkpoints/nq_sdpo-qwen3.6-35b-a3b-em-replay/` |
| Train replays | `res/train_replays/nq_sdpo-qwen3.6-35b-a3b-em-replay/` (`step_NNNN.jsonl`) |
| Train log | `exps/nq_sdpo-qwen3.6-35b-a3b-em-replay/train.log` |
| Experiment info | `exps/nq_sdpo-qwen3.6-35b-a3b-em-replay/info.json` |
| Launcher | `scripts/experiments/rui_meng/train_sdpo_v02_35b_a3b_replay.sh` |

Re-run:

```bash
cd /root/Search-R1
bash scripts/experiments/rui_meng/train_sdpo_v02_35b_a3b_replay.sh
```
