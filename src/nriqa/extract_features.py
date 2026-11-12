"""Extract frozen features for stable MANIQA / LIQE split probes."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from .config import resolve_device
from .data import build_transform, load_dataframe, make_loader


@torch.no_grad()
def extract(model_name: str, root: str, out: str, device: torch.device,
            batch_size: int = 8, dataset: str = "kadid10k"):
    import pyiqa

    df = load_dataframe(dataset, root)
    transform = build_transform(224, train=False, norm="none", eval_mode="center")
    loader = make_loader(df, transform, batch_size=batch_size, num_workers=8, shuffle=False, seed=42)
    metric = pyiqa.create_metric(model_name, device=device)
    net = metric.net.to(device).eval()

    feats, names = [], []
    for i, batch in enumerate(loader):
        x = batch["image"].to(device, non_blocking=True)
        if model_name == "maniqa":
            xn = (x - net.default_mean.to(x)) / net.default_std.to(x)
            net.vit(xn)
            f = net.extract_feature(net.save_output)
            net.save_output.outputs.clear()
            f = f.mean(dim=1)
        elif model_name == "liqe":
            xn = (x - net.default_mean.to(x)) / net.default_std.to(x)
            f = net.clip_model.encode_image(xn, pos_embedding=True)
        else:
            raise ValueError(model_name)
        feats.append(f.float().cpu().numpy())
        names.extend(list(batch["name"]))
        if i % 100 == 0:
            print(f"{i * batch_size}/{len(df)}", flush=True)
    X = np.concatenate(feats, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, features=X, names=np.asarray(names), mos=df["mos"].to_numpy(np.float32))
    print(f"saved {out} features={X.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["maniqa", "liqe"])
    ap.add_argument("--root", default="data/kadid10k")
    ap.add_argument("--dataset", default="kadid10k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    extract(args.model, args.root, args.out, resolve_device("cuda:0"), args.batch_size, args.dataset)


if __name__ == "__main__":
    main()
