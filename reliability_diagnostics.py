import pandas as pd
import numpy as np

# Load train transitions and raw events
tr = pd.read_parquet('data/splits/train_transitions_30min.parquet')
cols = ['user_id', 'video_id', 'time_ms', 'play_time_ms', 'duration_ms', 'long_view', 'is_like', 'is_forward', 'is_hate']
f1 = pd.read_csv('data/log_standard_4_08_to_4_21_1k.csv', usecols=cols + ['tab', 'is_click'])
f2 = pd.read_csv('data/log_standard_4_22_to_5_08_1k.csv', usecols=cols + ['tab', 'is_click'])
df = pd.concat([f1, f2], ignore_index=True)
df = df[(df['tab']==1) & (df['is_click']==1)].drop_duplicates(subset=['user_id', 'video_id', 'time_ms'])

# Calculate watch ratio
dur_pos = df['duration_ms'] > 0
wr = np.where(dur_pos, df['play_time_ms'] / df['duration_ms'], 0.5)
df['watch_ratio'] = np.clip(wr, 0.0, 1.0)

# Compute event weight (excluding is_click)
w = df['watch_ratio'] + 0.3 * df['is_like'] + 0.2 * df['is_forward'] - 0.5 * df['is_hate']
df['event_weight'] = np.clip(w, 0.1, 1.5)

print('Event weight distribution in Dataset A (tab=1 & click=1):')
print(df['event_weight'].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]))
min_clip = (df['event_weight'] <= 0.1).mean()
max_clip = (df['event_weight'] >= 1.5).mean()
print(f'Clipping at min (0.1): {min_clip:.4%}')
print(f'Clipping at max (1.5): {max_clip:.4%}')

# Signal correlations
sig_cols = ['watch_ratio', 'long_view', 'is_like', 'is_forward', 'is_hate', 'event_weight']
print('\nSignal Correlation Matrix (Dataset A):')
print(df[sig_cols].corr().round(4))

# Transition weights W1, W2, W3, W4 on training transitions
tr_sub = tr.head(20000).copy()
tr_sub = tr_sub.merge(df[['user_id', 'video_id', 'time_ms', 'event_weight']], 
                      left_on=['user_id', 'src_video_id', 'src_time_ms'], 
                      right_on=['user_id', 'video_id', 'time_ms'], how='left').rename(columns={'event_weight': 'w_src'})
tr_sub = tr_sub.merge(df[['user_id', 'video_id', 'time_ms', 'event_weight']], 
                      left_on=['user_id', 'tgt_video_id', 'tgt_time_ms'], 
                      right_on=['user_id', 'video_id', 'time_ms'], how='left').rename(columns={'event_weight': 'w_tgt'})

tr_sub['W1'] = 1.0
tr_sub['W2'] = tr_sub['w_src']
tr_sub['W3'] = tr_sub['w_tgt']
tr_sub['W4'] = np.sqrt(tr_sub['w_src'] * tr_sub['w_tgt'])

print('\nTransition Weights Summary (W1, W2, W3, W4):')
print(tr_sub[['W1', 'W2', 'W3', 'W4']].describe().round(4))
