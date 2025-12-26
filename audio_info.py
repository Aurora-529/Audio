#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Visualization Tool

This script visualizes audio files by displaying waveform and spectrogram in a single figure.
"""

import os
import sys
import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt


def load_audio(file_path: str) -> tuple:
    """
    Load audio file
    
    Args:
        file_path: Audio file path
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        # Try to load original channels
        y, sr = librosa.load(file_path, sr=None, mono=False)
        # Convert to mono for visualization if multi-channel
        if y.ndim > 1:
            y = librosa.to_mono(y)
    except Exception as e:
        # Fallback to mono loading
        y, sr = librosa.load(file_path, sr=None, mono=True)
    
    return y, sr


def plot_combined_visualization(y: np.ndarray, sr: int, file_name: str = ""):
    """
    Plot waveform and spectrogram in a single figure
    
    Args:
        y: Audio data
        sr: Sample rate
        file_name: File name for title display
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot waveform
    time = np.arange(len(y)) / sr
    ax1.plot(time, y)
    ax1.set_title(f'{file_name} - Waveform' if file_name else 'Waveform')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    # Plot spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr, cmap='viridis', ax=ax2)
    plt.colorbar(img, ax=ax2, format="%+2.0f dB")
    ax2.set_title(f'{file_name} - Spectrogram' if file_name else 'Spectrogram')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    
    return fig


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Audio Visualization Tool")
    parser.add_argument("--input", "-i", required=True, help="Input audio file path")
    parser.add_argument("--save", "-s", action="store_true", help="Save visualization to file")
    args = parser.parse_args()
    
    try:
        # Load audio
        y, sr = load_audio(args.input)
        file_name = os.path.splitext(os.path.basename(args.input))[0]
        
        # Plot combined visualization
        fig = plot_combined_visualization(y, sr, file_name)
        
        if args.save:
            # Save figure
            output_file = f"{file_name}_visualization.png"
            fig.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to: {output_file}")
            plt.close(fig)
        else:
            # Display figure
            plt.show()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

