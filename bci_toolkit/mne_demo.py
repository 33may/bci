from __future__ import annotations
import os
from .app import run_with_mne_file
from .data_loader import load_bci

def demo_mne():
    """Demo with BCI data using the load_bci function."""
    try:
        print("Loading BCI data...")
        raw_l, raw_r = load_bci(subject=1, runs=[6, 10])
        
        print(f"Left hand data: {raw_l.ch_names[:5]}... ({len(raw_l.ch_names)} channels)")
        print(f"Right hand data: {raw_r.ch_names[:5]}... ({len(raw_r.ch_names)} channels)")
        
        # Use left hand data for demo
        print("Using left hand data for visualization...")
        run_with_mne_file(raw_l)
        
    except Exception as e:
        print(f"Error loading BCI data: {e}")
        print("Falling back to local GDF files...")
        
        # Fallback to local GDF files
        data_dir = "data/BCICIV_2a_gdf/raw"
        gdf_files = []
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.gdf'):
                    gdf_files.append(os.path.join(data_dir, file))
        
        if not gdf_files:
            print(f"No GDF files found in {data_dir}")
            print("Please ensure the BCICIV_2a dataset is available")
            return
        
        # Use the first available file
        file_path = gdf_files[0]
        print(f"Using file: {file_path}")
        
        # Run the application
        run_with_mne_file(file_path)

if __name__ == "__main__":
    demo_mne()