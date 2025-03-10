
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0
# POLICY1_CONFIG=spi0_aloha_finetune_full
# POLICY1_DIR=/home/spirit-ai/codebase/PI_Official/checkpoints/shirtflatten_0225_27_bases2_full/29999
# POLICY1_PROMPT="fold the shirt"

# POLICY2_CONFIG=spi0_aloha_finetune_full_15steps
# POLICY2_DIR=/home/spirit-ai/codebase/PI_Official/checkpoints/spi0_aloha_15steps_quick_ftfull_0305/59999
# POLICY2_PROMPT="Fold the shirt"

# python scripts/serve_policy.py --default_prompt="${POLICY1_PROMPT}" policy:checkpoint --policy.config=$POLICY1_CONFIG --policy.dir=$POLICY1_DIR 
# python scripts/serve_policy.py --default_prompt="${POLICY2_PROMPT}" policy:checkpoint --policy.config=$POLICY2_CONFIG --policy.dir=$POLICY2_DIR 

POLICY1_CONFIG=spi0_aloha_finetune_flattenfold
POLICY1_DIR=/home/spirit-ai/codebase/PI_Official/checkpoints/spi0_aloha_flatten_fold_ftfull_0307_6w/55000
POLICY1_PROMPT="Flatten the shirt"
POLICY2_PROMPT="Fold the shirt"

python scripts/serve_multipolicy.py --default_prompt1="${POLICY1_PROMPT}" --default_prompt2="${POLICY2_PROMPT}" policy:checkpoint --policy.config=$POLICY1_CONFIG --policy.dir=$POLICY1_DIR




