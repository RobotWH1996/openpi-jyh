import argparse
import json
import numpy as np
from pathlib import Path

def compute_trajectory_quality(actions, lambda_jerk=0.01):
    """
    Computes heuristic quality of a trajectory based on actions/states.
    actions: numpy array of shape (T, action_dim)
    """
    T = len(actions)
    if T < 4:
        return 1.0, 1.0, 1.0 # Too short to evaluate, default to max
        
    # 1. Time Efficiency (Inverse of length)
    # We will normalize this across the dataset later
    time_score = 1.0 / T 
    
    # 2. Smoothness (Inverse of Jerk)
    # We compute derivatives to penalize jerky, shaky human teleop
    jerk_sum = 0
    path_length = 0
    
    for t in range(3, T):
        p0, p1, p2, p3 = actions[t-3:t+1]
        
        # Path length (L2 norm of step)
        path_length += np.linalg.norm(p3 - p2)
        
        # Jerk (3rd derivative of position, or 2nd if actions are velocities)
        v1, v2, v3 = p1-p0, p2-p1, p3-p2
        a1, a2 = v2-v1, v3-v2
        jerk = a2 - a1
        jerk_sum += np.linalg.norm(jerk)**2

    smooth_score = np.exp(-lambda_jerk * jerk_sum)
    path_score = 1.0 / (path_length + 1e-6)
    
    return time_score, smooth_score, path_score

def main():
    parser = argparse.ArgumentParser(description="Compute RWBC rewards for offline dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to your dataset")
    parser.add_argument("--output", type=str, default="rwbc_rewards.json", help="Output JSON file")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # NOTE: Change "*.npy" to match your dataset format (e.g., "*.hdf5" or "*.zarr")
    episodes = list(data_dir.rglob("*.npy")) 
    
    if not episodes:
        print(f"No episodes found in {data_dir}. Please adjust the file extension in the script.")
        return

    print(f"Found {len(episodes)} episodes. Computing heuristics...")
    
    raw_scores = {}
    time_scores, smooth_scores, path_scores = [], [], []
    
    for ep_path in episodes:
        # ---------------------------------------------------------
        # TODO: Adapt this loading logic to your specific dataset format
        # Example for numpy:
        # data = np.load(ep_path, allow_pickle=True).item()
        # actions = data["actions"]
        # ---------------------------------------------------------
        
        # Mocking actions here so the script runs out of the box
        actions = np.random.randn(100, 7) 
        
        t_s, s_s, p_s = compute_trajectory_quality(actions)
        raw_scores[str(ep_path)] = {"time": t_s, "smooth": s_s, "path": p_s}
        
        time_scores.append(t_s)
        smooth_scores.append(s_s)
        path_scores.append(p_s)
        
    # Normalize scores across the dataset so the BEST gets 1.0
    max_time = max(time_scores)
    max_smooth = max(smooth_scores)
    max_path = max(path_scores)
    
    final_rewards = {}
    for ep, scores in raw_scores.items():
        norm_time = scores["time"] / max_time
        norm_smooth = scores["smooth"] / max_smooth
        norm_path = scores["path"] / max_path
        
        # Weighted combination of heuristics (Tweak these weights if you care more about speed vs smoothness)
        final_reward = 0.4 * norm_time + 0.4 * norm_smooth + 0.2 * norm_path
        final_rewards[ep] = float(np.clip(final_reward, 0.1, 1.0))
        
    with open(args.output, "w") as f:
        json.dump(final_rewards, f, indent=4)
        
    print(f"Successfully saved {len(final_rewards)} reward weights to {args.output}")

if __name__ == "__main__":
    main()
