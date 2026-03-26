import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import os
import numpy as np
import torch

from cartpole_rs_switcher.training import train_switcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/switcher.pt")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--sigma", type=float, default=0.0, help="Gaussian noise std for RS augmentation (raw obs space)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    data = np.load(args.dataset)
    X = data["X"]
    y = data["y"]
    mean = data["state_mean"]
    std = data["state_std"]

    model = train_switcher(X, y, mean, std, hidden_dim=args.hidden_dim, epochs=args.epochs, sigma=args.sigma, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    torch.save({"state_dict": model.state_dict(), "obs_dim": X.shape[1], "hidden_dim": args.hidden_dim}, args.output)
    print(f"saved switcher to {args.output}")


if __name__ == "__main__":
    main()
