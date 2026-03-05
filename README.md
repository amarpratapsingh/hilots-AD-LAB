# Prototype: Minimal LiDAR Segmentation

This is a tiny, CPU-friendly prototype for point-wise LiDAR semantic segmentation. It uses a synthetic dataset and a small MLP model.

## Setup

Create a virtual environment if you want, then install requirements:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Train

python src/train.py --epochs 20 --num-points 1024 --batch-size 4

## Eval

python src/eval.py --checkpoint checkpoints/model.pt

## Notes

- This is a supervised single-branch prototype.
- Data is synthetic and generated on the fly for each sample.
- Default settings are tuned for low-end laptops.
