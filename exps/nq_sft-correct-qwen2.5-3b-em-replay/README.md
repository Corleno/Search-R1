# nq_sft-correct-qwen2.5-3b-em-replay

Online rejection-SFT (EM-correct trajectories + FIFO replay buffer) with **Qwen/Qwen2.5-3B**.

| Artifact | Path |
|----------|------|
| Checkpoints | `verl_checkpoints/nq_sft-correct-qwen2.5-3b-em-replay/` |
| Train replays (JSONL) | `res/train_replays/nq_sft-correct-qwen2.5-3b-em-replay/` (`step_NNNN.jsonl`) |
| Train log | `exps/nq_sft-correct-qwen2.5-3b-em-replay/train.log` |
| Experiment info | `exps/nq_sft-correct-qwen2.5-3b-em-replay/info.json` |
| Launcher | `scripts/experiments/rui_meng/train_sft_correct_v02_replay.sh` |

In-training FIFO replay buffer is **in-memory only** (capacity 4096, sample 256/step).

Re-run:

```bash
cd /root/Search-R1
bash scripts/experiments/rui_meng/train_sft_correct_v02_replay.sh
```
