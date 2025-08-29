import mne

mne.viz.set_browser_backend("qt")

raw = mne.io.read_raw_gdf("/Users/may/projects/bci/data/BCICIV_2a_gdf/A01T.gdf", preload=True)

raw.plot(duration=10, block=True)