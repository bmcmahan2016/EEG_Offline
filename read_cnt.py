import mne

raw = mne.io.read_raw_cnt('data/cnt_data.cnt')
print(raw.info)