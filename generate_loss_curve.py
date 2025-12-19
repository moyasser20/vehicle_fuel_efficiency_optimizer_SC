#!/usr/bin/env python3
"""
Generate Training Loss Curve for Vehicle Fuel Efficiency Model
This script creates a visualization of the training loss over epochs.
"""

import matplotlib.pyplot as plt
import numpy as np

# Loss values extracted from training output
# From the output: Epoch 1, 2, 3, 4, 5, 11, 21, 31, 41, 51, 61, 71, 81, 91, 96, 97, 98, 99, 100
known_losses = {
    1: 0.003880,
    2: 0.001334,
    3: 0.000963,
    4: 0.000805,
    5: 0.000719,
    11: 0.000552,
    21: 0.000450,
    31: 0.000411,
    41: 0.000369,
    51: 0.000348,
    61: 0.000341,
    71: 0.000327,
    81: 0.000317,
    91: 0.000312,
    96: 0.000302,
    97: 0.000305,
    98: 0.000309,
    99: 0.000304,
    100: 0.000309
}

# Interpolate missing epochs using linear interpolation
epochs = list(range(1, 101))
losses = []

for epoch in epochs:
    if epoch in known_losses:
        losses.append(known_losses[epoch])
    else:
        # Find the two nearest known epochs
        prev_epoch = max([e for e in known_losses.keys() if e < epoch], default=1)
        next_epoch = min([e for e in known_losses.keys() if e > epoch], default=100)
        
        # Linear interpolation
        prev_loss = known_losses[prev_epoch]
        next_loss = known_losses[next_epoch]
        
        # Interpolate
        ratio = (epoch - prev_epoch) / (next_epoch - prev_epoch)
        interpolated_loss = prev_loss + (next_loss - prev_loss) * ratio
        losses.append(interpolated_loss)

# Convert to numpy array
epochs_array = np.array(epochs)
losses_array = np.array(losses)

# Create the plot
plt.figure(figsize=(12, 7))

# Plot the loss curve
plt.plot(epochs_array, losses_array, 'b-', linewidth=2.5, label='Training Loss (MSE)', marker='o', markersize=4, markevery=10)

# Add grid
plt.grid(True, alpha=0.3, linestyle='--')

# Labels and title
plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('MSE Loss', fontsize=14, fontweight='bold')
plt.title('Training Loss Curve - Vehicle Fuel Efficiency Model', fontsize=16, fontweight='bold', pad=20)

# Add legend
plt.legend(fontsize=12, loc='upper right')

# Add annotations for key points
plt.annotate(f'Initial Loss: {known_losses[1]:.6f}', 
             xy=(1, known_losses[1]), 
             xytext=(15, known_losses[1] + 0.0005),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10, color='red', fontweight='bold')

plt.annotate(f'Final Loss: {known_losses[100]:.6f}', 
             xy=(100, known_losses[100]), 
             xytext=(70, known_losses[100] + 0.0005),
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
             fontsize=10, color='green', fontweight='bold')

# Set axis limits for better visualization
plt.xlim(0, 105)
plt.ylim(min(losses_array) * 0.8, max(losses_array) * 1.2)

# Add text box with summary statistics
textstr = f'Training Summary:\n'
textstr += f'Initial Loss: {known_losses[1]:.6f}\n'
textstr += f'Final Loss: {known_losses[100]:.6f}\n'
textstr += f'Reduction: {((known_losses[1] - known_losses[100]) / known_losses[1] * 100):.1f}%\n'
textstr += f'Total Epochs: 100'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props, family='monospace')

# Adjust layout
plt.tight_layout()

# Save the figure
output_filename = 'training_loss_curve.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Training loss curve saved as: {output_filename}")

# Also save as PDF for high quality
output_pdf = 'training_loss_curve.pdf'
plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
print(f"[OK] Training loss curve saved as: {output_pdf}")

# Display the plot
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("Training Loss Summary")
print("="*60)
print(f"Initial Loss (Epoch 1):  {known_losses[1]:.6f}")
print(f"Final Loss (Epoch 100):  {known_losses[100]:.6f}")
print(f"Loss Reduction:           {((known_losses[1] - known_losses[100]) / known_losses[1] * 100):.2f}%")
print(f"Average Loss:             {np.mean(losses_array):.6f}")
print(f"Min Loss:                 {np.min(losses_array):.6f}")
print(f"Max Loss:                 {np.max(losses_array):.6f}")
print("="*60)
