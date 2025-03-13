export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

python scripts/serve_policy.py --env SPI0_ARX_FULL --default_prompt='Fold the shirt'
