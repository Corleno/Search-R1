# Usage: bash eval_all_step.sh
# Example: bash scripts/experiments/rui_meng/eval_replay_sdpo_v02_noclip/eval_all_step.sh

# eval the step 1 to 200 every 10 steps, step is padded to 4 digits
for step in {0001..0200..10}; do
    bash scripts/experiments/rui_meng/eval_replay_sdpo_v02_noclip/eval_step_n.sh ${step}
done