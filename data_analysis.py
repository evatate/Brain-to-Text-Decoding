import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# PATHS/CONFIGURATION
# ------------------------------
data_root = "data/hdf5_data_final"
splits = ["train", "val", "test"]
brain_areas = ["ventral_6v", "area_4", "55b", "dorsal_6v"]
feature_types = ["threshold_crossings", "spike_band_power"]

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------

def load_h5py_file(file_path):
    data = {
        'neural_features': [],
        'n_time_steps': [],
        'seq_class_ids': [],
        'seq_len': [],
        'transcriptions': [],
        'sentence_label': [],
        'session': [],
        'block_num': [],
        'trial_num': [],
    }

    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            g = f[key]

            neural_features = g['input_features'][:] 
            n_time_steps = g.attrs['n_time_steps']
            if 'seq_class_ids' in g:
                seq_class_ids = g['seq_class_ids'][:]
                seq_class_ids = seq_class_ids.astype(int).tolist() if len(seq_class_ids) > 0 else []
            else:
                seq_class_ids = []
            seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else len(seq_class_ids)

            # handles both bytes and string arrays
            transcription = None
            if 'transcription' in g:
                t = g['transcription'][()]
                transcription = t.decode('ascii') if isinstance(t, bytes) else str(t)

            sentence_label = None
            if 'sentence_label' in g.attrs:
                s = g.attrs['sentence_label']
                sentence_label = s.decode('ascii') if isinstance(s, bytes) else str(s)

            session = g.attrs['session']
            block_num = g.attrs['block_num']
            trial_num = g.attrs['trial_num']

            data['neural_features'].append(neural_features)
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(seq_class_ids)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['session'].append(session)
            data['block_num'].append(block_num)
            data['trial_num'].append(trial_num)

    return data

def summarize_neural_features(neural_list):
    """Compute mean and variance per feature across all valid trials"""
    # flatten across trials and time steps
    valid_trials = [t for t in neural_list if t.size > 0]
    if not valid_trials:
        return np.nan, np.nan

    stacked = np.concatenate(valid_trials, axis=0)
    mean_feats = np.mean(stacked, axis=0)
    var_feats = np.var(stacked, axis=0)
    return mean_feats, var_feats

def get_area_feature_indices():
    """Return dictionary mapping brain areas to their feature indices"""
    idx = {}
    for i, area in enumerate(brain_areas):
        tc_start = i*64
        tc_end = tc_start + 64
        sb_start = 256 + i*64
        sb_end = sb_start + 64
        idx[area] = {
            "threshold_crossings": (tc_start, tc_end),
            "spike_band_power": (sb_start, sb_end)
        }
    return idx

# ------------------------------
# AGGREGATE DATA
# ------------------------------

all_stats = []

area_indices = get_area_feature_indices()

for session_folder in sorted(os.listdir(data_root)):
    session_path = os.path.join(data_root, session_folder)
    if not os.path.isdir(session_path):
        continue

    for split in splits:
        h5_file = os.path.join(session_path, f"data_{split}.hdf5")
        if not os.path.exists(h5_file):
            continue
        
        # load data using custom loader
        data = load_h5py_file(h5_file)
        neural_list = data['neural_features']  # list of trials, each [T, features]
        
        if not neural_list:
            continue  # skip if no trials

        # compute total trials, features, and time steps
        n_trials = len(neural_list)
        n_features = max(trial.shape[1] for trial in neural_list if trial.size > 0)
        total_time = sum(trial.shape[0] for trial in neural_list if trial.size > 0)

        print(f"Processing session: {session_folder} | Split: {split} | "
              f"Trials: {n_trials} | Features: {n_features} | Total time steps: {total_time}")

        # compute mean and variance per trial safely
        trial_means = []
        trial_vars = []
        for trial in neural_list:
            if trial.size == 0:
                continue
            trial_means.append(np.mean(trial, axis=0))
            trial_vars.append(np.var(trial, axis=0))
        mean_feats = np.mean(trial_means, axis=0) if trial_means else np.nan
        var_feats = np.mean(trial_vars, axis=0) if trial_vars else np.nan

        # compute area-wise stats
        for area in brain_areas:
            for ftype in feature_types:
                start, end = area_indices[area][ftype]

                # collect only valid trials for this area
                area_data_list = []
                for trial in neural_list:
                    if trial.size == 0 or trial.shape[1] < end:
                        continue
                    area_data_list.append(trial[:, start:end])

                if not area_data_list:
                    mean_val, var_val = np.nan, np.nan
                else:
                    area_vals = np.concatenate(area_data_list, axis=0)
                    mean_val = np.mean(area_vals)
                    var_val = np.var(area_vals)

                all_stats.append({
                    "session": session_folder,
                    "split": split,
                    "area": area,
                    "feature_type": ftype,
                    "mean": mean_val,
                    "variance": var_val
                })

        # sentence/phoneme lengths for train/val splits
        if split != "test":
            # filter out empty strings or empty arrays
            sent_lengths = [len(s) for s in data['sentence_label'] if s]
            phon_lengths = [len(seq) for seq in data['seq_class_ids'] if seq is not None and len(seq) > 0]
            avg_sent_len = np.mean(sent_lengths) if sent_lengths else np.nan
            avg_phon_len = np.mean(phon_lengths) if phon_lengths else np.nan


# ------------------------------
# CREATE DATAFRAME
# ------------------------------
stats_df = pd.DataFrame(all_stats)

# ------------------------------
# VISUALIZATIONS
# ------------------------------
# variance across trials
variance_by_area = stats_df.groupby(['area', 'feature_type'])['variance'].mean().reset_index()
plt.figure(figsize=(12,6))
sns.barplot(data=variance_by_area, x='area', y='variance', hue='feature_type')
plt.title("Average Variance by Brain Area and Feature Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# number of trials per session/split
trials_per_session = stats_df.groupby(['session', 'split'])['mean'].count().reset_index()
plt.figure(figsize=(12,6))
sns.barplot(data=trials_per_session, x='session', y='mean', hue='split')
plt.ylabel("Number of Area-Feature Entries (proxy for trials)")
plt.xticks(rotation=45)
plt.title("Number of Trials per Session and Split")
plt.tight_layout()
plt.show()

# average sentence and phoneme lengths
sent_lengths = [len(s) for s in data['sentence_label'] if s]
phon_lengths = [len(seq) for seq in data['seq_class_ids'] if seq is not None]

avg_sent_len = np.mean(sent_lengths) if sent_lengths else np.nan
avg_phon_len = np.mean(phon_lengths) if phon_lengths else np.nan
print(f"Average sentence length: {avg_sent_len:.2f}")
print(f"Average phoneme sequence length: {avg_phon_len:.2f}")

# correlation between areas
area_means = stats_df.pivot_table(index='feature_type', columns='area', values='mean')
corr_matrix = area_means.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation of Mean Features Across Brain Areas")
plt.tight_layout()
plt.show()

# trial time-step distribution
time_steps = [trial.shape[0] for trial in neural_list if trial.size > 0]
plt.figure(figsize=(10,5))
sns.histplot(time_steps, bins=20, kde=True)
plt.xlabel("Time Steps per Trial")
plt.ylabel("Count")
plt.title("Distribution of Trial Lengths")
plt.tight_layout()
plt.show()

# feature type comparison
plt.figure(figsize=(12,6))
sns.barplot(data=stats_df, x='area', y='mean', hue='feature_type')
plt.xticks(rotation=45)
plt.title("Mean Feature Value by Brain Area and Feature Type")
plt.tight_layout()
plt.show()

# identify outliers
outliers = stats_df[(stats_df['mean'] > stats_df['mean'].quantile(0.99)) |
                    (stats_df['mean'] < stats_df['mean'].quantile(0.01))]
print("Potential outlier area-feature entries:")
print(outliers)