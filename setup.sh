# usage: source setup.sh

# setup uv environment (should be enough for Wan and SAM2 steps)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync
source .venv/bin/activate

# get TraceAnything checkpoint
mkdir -p TraceAnything/checkpoints
curl -L -o TraceAnything/checkpoints/trace_anything.pt https://huggingface.co/depth-anything/trace-anything/resolve/main/trace_anything.pt?download=true

# place our registration script into the correct directory
mv register_pointclouds.py third_party/DiffusionReg

# more dependencies
sudo apt-get update
sudo apt-get install -y libgl1