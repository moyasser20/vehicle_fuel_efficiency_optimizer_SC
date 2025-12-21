#!/usr/bin/env python3
"""
Extract Loss Values from Java Output and Generate Training Loss Curve
This script can parse Java console output to extract loss values automatically.
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import sys

def extract_losses_from_text(text):
    """Extract loss values from Java output text."""
    # Pattern to match: "Epoch X: 0.XXXXXX" or "Epoch X/100 - Average Loss: 0.XXXXXX"
    patterns = [
        r'Epoch\s+(\d+):\s+([\d.]+)',
        r'Epoch\s+(\d+)/\d+\s+-\s+Average\s+Loss:\s+([\d.]+)',
    ]
    
    losses = {}
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for epoch, loss in matches:
            losses[int(epoch)] = float(loss)
    
    return losses

def interpolate_losses(known_losses, total_epochs=100):
    """Interpolate missing epochs using linear interpolation."""
    epochs = list(range(1, total_epochs + 1))
    losses = []
    
    for epoch in epochs:
        if epoch in known_losses:
            losses.append(known_losses[epoch])
        else:
            # Find the two nearest known epochs
            prev_epoch = max([e for e in known_losses.keys() if e < epoch], default=1)
            next_epoch = min([e for e in known_losses.keys() if e > epoch], default=total_epochs)
            
            # Linear interpolation
            prev_loss = known_losses[prev_epoch]
            next_loss = known_losses[next_epoch]
            
            # Interpolate
            if next_epoch != prev_epoch:
                ratio = (epoch - prev_epoch) / (next_epoch - prev_epoch)
                interpolated_loss = prev_loss + (next_loss - prev_loss) * ratio
            else:
                interpolated_loss = prev_loss
            losses.append(interpolated_loss)
    
    return epochs, losses

def plot_loss_curve(epochs, losses, output_filename='training_loss_curve.png'):
    """Generate and save the training loss curve."""
    epochs_array = np.array(epochs)
    losses_array = np.array(losses)
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot the loss curve
    plt.plot(epochs_array, losses_array, 'b-', linewidth=2.5, 
             label='Training Loss (MSE)', marker='o', markersize=4, markevery=10)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Labels and title
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('MSE Loss', fontsize=14, fontweight='bold')
    plt.title('Training Loss Curve - Vehicle Fuel Efficiency Model', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    plt.legend(fontsize=12, loc='upper right')
    
    # Add annotations for key points
    initial_loss = losses[0]
    final_loss = losses[-1]
    
    plt.annotate(f'Initial Loss: {initial_loss:.6f}', 
                 xy=(1, initial_loss), 
                 xytext=(15, initial_loss * 1.3),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                 fontsize=10, color='red', fontweight='bold')
    
    plt.annotate(f'Final Loss: {final_loss:.6f}', 
                 xy=(len(epochs), final_loss), 
                 xytext=(len(epochs) - 30, final_loss * 1.3),
                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                 fontsize=10, color='green', fontweight='bold')
    
    # Set axis limits
    plt.xlim(0, len(epochs) + 5)
    plt.ylim(min(losses_array) * 0.8, max(losses_array) * 1.2)
    
    # Add text box with summary statistics
    reduction = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
    textstr = f'Training Summary:\n'
    textstr += f'Initial Loss: {initial_loss:.6f}\n'
    textstr += f'Final Loss: {final_loss:.6f}\n'
    textstr += f'Reduction: {reduction:.1f}%\n'
    textstr += f'Total Epochs: {len(epochs)}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props, family='monospace')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Training loss curve saved as: {output_filename}")
    
    # Also save as PDF
    pdf_filename = output_filename.replace('.png', '.pdf')
    plt.savefig(pdf_filename, bbox_inches='tight', facecolor='white')
    print(f"[OK] Training loss curve saved as: {pdf_filename}")
    
    # Display the plot
    plt.show()
    
    return initial_loss, final_loss, reduction

def main():
    # Loss values from the provided output
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
    
    # Interpolate missing epochs
    epochs, losses = interpolate_losses(known_losses, total_epochs=100)
    
    # Generate plot
    initial_loss, final_loss, reduction = plot_loss_curve(epochs, losses)
    
    # Print summary
    print("\n" + "="*60)
    print("Training Loss Summary")
    print("="*60)
    print(f"Initial Loss (Epoch 1):  {initial_loss:.6f}")
    print(f"Final Loss (Epoch 100):  {final_loss:.6f}")
    print(f"Loss Reduction:           {reduction:.2f}%")
    print(f"Average Loss:             {np.mean(losses):.6f}")
    print(f"Min Loss:                 {np.min(losses):.6f}")
    print(f"Max Loss:                 {np.max(losses):.6f}")
    print("="*60)
    
    # Optionally, if text file provided, extract from it
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
            extracted_losses = extract_losses_from_text(text)
            if extracted_losses:
                print(f"\n[INFO] Extracted {len(extracted_losses)} loss values from {filename}")
                epochs2, losses2 = interpolate_losses(extracted_losses, total_epochs=100)
                plot_loss_curve(epochs2, losses2, 'training_loss_curve_from_file.png')
        except Exception as e:
            print(f"[WARNING] Could not read file {filename}: {e}")

if __name__ == '__main__':
    main()

