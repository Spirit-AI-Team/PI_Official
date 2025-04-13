# To start inference, you need to run three terminals in total

## open first terminal:
```bash
    cd codebase/ARX_R5/ARX_CAN/
    ./can.sh
```

## open second terminal:
```bash
    cd ~/codebase/PI_Official
    source .venv/bin/activate
    python scripts/serve_policy.py --env SPI0_ARX_FULL3 --default_prompt='Flatten the shirt' 
    # The env and default prompt settings depend on your task
```

## open second terminal:
```bash
    cd ~/codebase/PI_Official
    su #pswd: ask lky or wwx
    cd examples/arx_r5_real_teleop/bimanual/
    source setup.sh
    cd ..
    source .venv/bin/activate
    cd ../..
    # Now you are ready to go! Run the following command to reset the arms
    python examples/arx_r5_real_teleop/arx_gohome.py
    # Start inference episode
    python examples/arx_r5_real_teleop/main.py
```