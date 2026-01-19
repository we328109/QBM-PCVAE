import argparse
import torch
from model import PILP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dim", type=int, default=249)
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = PILP(
        ft_dim=args.traj_dim,
        latent_size=args.latent_size,
    ).to(device)
    model.train()

    features = torch.randn(args.batch_size, args.traj_dim, device=device)
    ground_truth = torch.randn(args.batch_size, 7, device=device)
    crystal_gt = torch.randint(0, 14, (args.batch_size, 1), device=device)

    crystal, prediction, q_logits, zeta, kl_loss = model(
        ground_truth,
        features,
        crystal_gt,
    )

    loss = kl_loss + crystal.mean() + sum(p.mean() for p in prediction)
    loss.backward()

    print("Sanity check complete.")
    print(f"KL loss: {kl_loss.item():.6f}")


if __name__ == "__main__":
    main()
