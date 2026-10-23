"""Generate training and evaluation figures for Robot Navigation.

Run from the project root:
    python -m src.visualize
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

LOG_DIR = Path('results/logs')
PLOT_DIR = Path('results/plots')
PLOT_DIR.mkdir(parents=True, exist_ok=True)
WINDOW = 100

EXPERIMENTS = {
    'Q-Learning': 'q_learning_training.csv',
    'DQN (4 features)': 'dqn_training.csv',
    'Double DQN (4 features)': 'double_dqn_training.csv',
    'Double DQN (8 features)': 'double_dqn_8feat_training.csv',
    'Double DQN (8 features, simple reward)': 'double_dqn_simple_reward_training.csv',
    'Double DQN (12 features + history)': 'double_dqn_history5_training.csv',
}

def load_logs():
    logs = {}
    for label, filename in EXPERIMENTS.items():
        path = LOG_DIR / filename
        if not path.exists():
            print(f'Skipping missing file: {path}')
            continue
        df = pd.read_csv(path)
        if 'successes' in df.columns:
            df['success_rate'] = df['successes'].astype(float) * 100
        elif 'success' in df.columns:
            df['success_rate'] = df['success'].astype(float) * 100
        else:
            print(f'Skipping {path}: missing success column')
            continue
        logs[label] = df
    return logs

def plot_metric(logs, column, title, ylabel, filename, labels=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, df in logs.items():
        if labels is not None and label not in labels:
            continue
        if column not in df.columns:
            continue
        smoothed = df[column].rolling(WINDOW, min_periods=1).mean()
        ax.plot(df['episode'], smoothed, label=label, linewidth=1.8)
    ax.set(title=title, xlabel='Training episode', ylabel=ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=180)
    plt.close(fig)
    print(f'Saved: {PLOT_DIR / filename}')

def plot_penalty_sweep():
    df = pd.DataFrame({
        'lambda': [0, .01, .05, .1, .25, .5, 1.0],
        'success': [79.2, 80.8, 82.2, 85.6, 92.2, 94.0, 94.2],
        'steps': [50.15, 47.22, 44.63, 38.34, 26.59, 23.12, 22.76],
        'efficiency': [96.10, 95.83, 95.51, 94.80, 93.48, 93.38, 93.14],
        'optimal': [329, 329, 329, 332, 339, 344, 344],
    })
    # These are the user's reported 500-episode results.
    metrics = [
        ('success', 'Success rate (%)', 'penalty_success.png'),
        ('steps', 'Mean episode steps', 'penalty_steps.png'),
        ('efficiency', 'Mean path efficiency (%)', 'penalty_efficiency.png'),
        ('optimal', 'Optimal paths (of 500)', 'penalty_optimal_paths.png'),
    ]
    for col, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df['lambda'], df[col], marker='o', linewidth=2)
        ax.set(xlabel='Revisit penalty λ', ylabel=ylabel, title=ylabel + ' vs. λ')
        ax.set_xticks(df['lambda'])
        ax.set_xticklabels([f'{x:g}' for x in df['lambda']])
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / filename, dpi=180)
        plt.close(fig)
        print(f'Saved: {PLOT_DIR / filename}')

def main():
    logs = load_logs()
    # Compare algorithms on success and episode length; reward functions vary.
    for col, title, ylabel, filename in [
        ('success_rate', 'Training success rate (100-episode rolling mean)', 'Success rate (%)', 'training_success.png'),
        ('steps', 'Training episode length (100-episode rolling mean)', 'Steps', 'training_steps.png'),
        ('collisions', 'Training collisions (100-episode rolling mean)', 'Collisions per episode', 'training_collisions.png'),
    ]:
        plot_metric(logs, col, title, ylabel, filename)
    # Reward plot restricted to shaped-reward Double DQN variants.
    comparable = ['Double DQN (4 features)', 'Double DQN (8 features)', 'Double DQN (12 features + history)']
    plot_metric(logs, 'reward', 'Double DQN training reward (shaped reward)', 'Reward', 'double_dqn_rewards.png', comparable)
    plot_penalty_sweep()

if __name__ == '__main__':
    main()
