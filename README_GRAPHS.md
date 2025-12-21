# Generating Training Loss Curves

This directory contains Python scripts to generate training loss curve visualizations for the Vehicle Fuel Efficiency Model.

## Files

1. **`generate_loss_curve.py`** - Main script with hardcoded loss values
2. **`extract_and_plot_loss.py`** - Advanced script that can extract losses from text files
3. **`training_loss_curve.png`** - Generated PNG image (300 DPI)
4. **`training_loss_curve.pdf`** - Generated PDF (high quality)

## Quick Start

### Option 1: Use Pre-configured Script

Simply run:
```bash
python generate_loss_curve.py
```

This will:
- Generate `training_loss_curve.png` (300 DPI)
- Generate `training_loss_curve.pdf` (high quality)
- Display the graph
- Print summary statistics

### Option 2: Extract from Java Output

If you have the Java output saved in a text file:

```bash
python extract_and_plot_loss.py output.txt
```

The script will automatically:
- Extract loss values from the text
- Interpolate missing epochs
- Generate the graph

## Requirements

Install required packages:
```bash
pip install matplotlib numpy
```

## Output

The scripts generate:

1. **PNG Image** (`training_loss_curve.png`)
   - 300 DPI resolution
   - Suitable for reports and presentations
   - File size: ~200-500 KB

2. **PDF** (`training_loss_curve.pdf`)
   - Vector format (scalable)
   - Best for printing
   - File size: ~50-100 KB

## Graph Features

The generated graph includes:
- ✅ Training loss curve over 100 epochs
- ✅ Grid for easy reading
- ✅ Annotations for initial and final loss
- ✅ Summary statistics box
- ✅ Professional styling

## Example Output

```
[OK] Training loss curve saved as: training_loss_curve.png
[OK] Training loss curve saved as: training_loss_curve.pdf

============================================================
Training Loss Summary
============================================================
Initial Loss (Epoch 1):  0.003880
Final Loss (Epoch 100):  0.000309
Loss Reduction:           92.04%
Average Loss:             0.000440
Min Loss:                 0.000302
Max Loss:                 0.003880
============================================================
```

## Loss Values Used

From your training output:
- Epoch 1: 0.003880
- Epoch 2: 0.001334
- Epoch 3: 0.000963
- Epoch 4: 0.000805
- Epoch 5: 0.000719
- Epoch 11: 0.000552
- Epoch 21: 0.000450
- Epoch 31: 0.000411
- Epoch 41: 0.000369
- Epoch 51: 0.000348
- Epoch 61: 0.000341
- Epoch 71: 0.000327
- Epoch 81: 0.000317
- Epoch 91: 0.000312
- Epoch 96: 0.000302
- Epoch 97: 0.000305
- Epoch 98: 0.000309
- Epoch 99: 0.000304
- Epoch 100: 0.000309

Missing epochs are interpolated using linear interpolation.

## Customization

To modify the graph, edit `generate_loss_curve.py`:

- Change figure size: `plt.figure(figsize=(width, height))`
- Change colors: Modify `'b-'` (blue line)
- Change markers: Modify `marker='o'` and `markersize=4`
- Change DPI: Modify `dpi=300` in `savefig()`

## Troubleshooting

**Error: "ModuleNotFoundError: No module named 'matplotlib'"**
```bash
pip install matplotlib numpy
```

**Graph doesn't display:**
- The script saves files even if display fails
- Check that `training_loss_curve.png` was created
- On headless systems, graphs are saved but not displayed

**Encoding errors:**
- The script uses ASCII-safe characters
- If issues persist, ensure your terminal supports UTF-8

## For Submission

Include in your ZIP file:
- ✅ `generate_loss_curve.py` (Python script)
- ✅ `training_loss_curve.png` (Generated graph)
- ✅ `training_loss_curve.pdf` (Optional, high quality version)

These files demonstrate that you can generate graphs supporting your results (Requirement 6).

