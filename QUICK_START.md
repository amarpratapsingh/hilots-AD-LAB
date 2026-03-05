# Quick Reference Guide - Enhanced Prototype

## 🎯 One-Command Quick Start

```bash
# Test everything works
cd /home/amar/Documents/HiLoTs/prototype
/home/amar/Documents/HiLoTs/.venv/bin/python src/test_model.py

# Train a model (5 min)
/home/amar/Documents/HiLoTs/.venv/bin/python src/train.py --epochs 10 --train-samples 100 --cpu

# Evaluate the model  
/home/amar/Documents/HiLoTs/.venv/bin/python src/eval.py --checkpoint checkpoints/model.pt --cpu

# Run inference
/home/amar/Documents/HiLoTs/.venv/bin/python src/infer.py --checkpoint checkpoints/model.pt --cpu
```

## 📊 What You Get

### After Training:
- `checkpoints/model.pt` - Final model
- `checkpoints/model_best.pt` - Best performing model
- Console output with per-epoch metrics

### After Evaluation:
- Overall accuracy and mIoU
- Per-class IoU, precision, recall, F1-score
- Confusion matrix
- Detailed metrics report

### After Inference:
- `predictions.npy` - Class predictions for each point
- Class distribution statistics

## 🔧 Key Files

```
prototype/src/
├── train.py          # Main training script
├── eval.py           # Evaluation script  
├── infer.py          # Inference script
├── test_model.py     # Unit tests (run this first!)
├── data/
│   ├── synthetic.py      # Dataset: 5 classes, 11 features
│   └── transforms.py     # Augmentation (ready to use)
├── models/
│   └── point_mlp.py      # PointMLP: 69K params
└── utils/
    ├── config.py         # Configuration system
    ├── metrics.py        # IoU, accuracy
    └── seed.py           # Reproducibility
```

## 🎨 Customization Examples

### Change Number of Classes
```bash
python src/train.py --num-classes 10
```

### Bigger Model
Edit `src/train.py` line 64:
```python
hidden_dims=[256, 512, 256]  # instead of [128, 256, 128]
```

### More Training Data
```bash
python src/train.py --train-samples 500 --val-samples 100
```

### Use GPU
```bash
python src/train.py --epochs 20  # Remove --cpu flag
```

## 📈 Performance Expectations

**On synthetic data (random labels):**
- Expected mIoU: 0.08-0.15 (baseline for 5 classes)
- Expected accuracy: 0.20-0.25 (5 classes = 20% random baseline)

**On real data (with actual patterns):**
- Expected mIoU: 0.40-0.70 (depends on dataset)
- Expected accuracy: 0.60-0.85 (depends on dataset)

## 🚀 Next Development Steps

1. **Add Real Data**
   - Create `src/data/kitti_dataset.py`
   - Load real point clouds and labels
   - Replace synthetic data in train.py

2. **Improve Model**
   - Add PointNet++ layers
   - Increase model capacity
   - Add attention mechanisms

3. **Better Training**
   - Enable data augmentation (transforms already implemented!)
   - Add more advanced schedulers
   - Implement early stopping

## ✅ Verification Checklist

- [x] `test_model.py` passes all tests
- [x] `train.py` completes without errors
- [x] `eval.py` produces detailed metrics
- [x] `infer.py` generates predictions
- [x] Checkpoints saved correctly
- [x] Configuration system works
- [x] All 11 feature channels used
- [x] 5 classes supported

## 📚 Documentation

- `README_ENHANCED.md` - Full documentation  
- `IMPLEMENTATION_SUMMARY.md` - What was built
- This file - Quick reference

## 💡 Tips

1. **Start small**: Use `--train-samples 20 --epochs 3` for quick testing
2. **Monitor metrics**: Loss should decrease, mIoU should increase
3. **Best model**: Use `model_best.pt` for evaluation (highest mIoU)
4. **CPU vs GPU**: Add `--cpu` flag if no GPU available
5. **Reproducibility**: Same `--seed` gives same results

## 🐛 Troubleshooting

**Out of memory?**
```bash
python src/train.py --batch-size 2 --train-samples 50
```

**Training too slow?**
```bash
python src/train.py --num-points 512  # Reduce points per sample
```

**Want faster iteration?**
```bash
python src/train.py --epochs 3 --train-samples 20 --cpu
```

## 🎉 You now have a complete working model!

Everything is tested, documented, and ready to use. Enjoy! 🚀
