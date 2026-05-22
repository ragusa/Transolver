import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import torch
from torch.utils.data import DataLoader

from heat2d_dataset import Heat2DDataset, heat2d_batch_size_one_collate
from model.Transolver_Irregular_Mesh import Model


def build_parser():
    parser = argparse.ArgumentParser("Heat2D Transolver experiment")
    parser.add_argument("--mode", type=str, default="overfit", choices=["smoke", "overfit", "train", "eval"])
    parser.add_argument("--data_path", type=str, nargs="+", default=["fom_generation/data/heat2d_pilot"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--sample-name", type=str, default=None)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--include-boundary-mask", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--n-hidden", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=int, default=1)
    parser.add_argument("--fun-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--slice_num", type=int, default=16)
    parser.add_argument("--ref", type=int, default=8)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="results/heat2d_train_small")
    parser.add_argument("--plot-dir", type=str, default="results/heat2d_overfit")
    parser.add_argument("--plot-samples", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval-output-dir", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def get_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return torch.device(name)


def compute_stats(dataset):
    features = []
    targets = []
    for item in dataset:
        features.append(item["node_features"])
        targets.append(item["target"])
    feature_values = torch.cat(features, dim=0)
    target_values = torch.cat(targets, dim=0)
    return {
        "feature_mean": feature_values.mean(dim=0, keepdim=True),
        "feature_std": feature_values.std(dim=0, keepdim=True).clamp_min(1e-8),
        "target_mean": target_values.mean(dim=0, keepdim=True),
        "target_std": target_values.std(dim=0, keepdim=True).clamp_min(1e-8),
    }


def load_stats_from_checkpoint(checkpoint):
    stats = checkpoint["normalization_stats"]
    return {key: value.detach().cpu() for key, value in stats.items()}


def stats_to_jsonable(stats):
    return {key: value.detach().cpu().tolist() for key, value in stats.items()}


def normalize_batch(batch, stats, device, normalize=True):
    pos = batch["pos"].to(device)
    fx = batch["node_features"].to(device)
    y = batch["target"].to(device)

    if not normalize:
        return pos, fx, y

    feature_mean = stats["feature_mean"].to(device)
    feature_std = stats["feature_std"].to(device)
    target_mean = stats["target_mean"].to(device)
    target_std = stats["target_std"].to(device)
    fx = (fx - feature_mean.unsqueeze(0)) / feature_std.unsqueeze(0)
    y = (y - target_mean.unsqueeze(0)) / target_std.unsqueeze(0)
    return pos, fx, y


def decode_target(y, stats, normalize=True):
    if not normalize:
        return y
    target_mean = stats["target_mean"].to(y.device)
    target_std = stats["target_std"].to(y.device)
    return y * target_std.unsqueeze(0) + target_mean.unsqueeze(0)


def plot_field(path, pos, triangles, values, title, cmap="coolwarm"):
    path.parent.mkdir(parents=True, exist_ok=True)
    triangulation = mtri.Triangulation(pos[:, 0], pos[:, 1], triangles)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    color = ax.tripcolor(triangulation, values, shading="gouraud", cmap=cmap)
    fig.colorbar(color, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_overfit_plots(batch, pred, target, plot_dir):
    plot_dir = Path(plot_dir)
    pos = batch["pos"][0].detach().cpu().numpy()
    triangles = batch["triangles"].detach().cpu().numpy()
    pred_np = pred[0, :, 0].detach().cpu().numpy()
    target_np = target[0, :, 0].detach().cpu().numpy()
    abs_error = abs(pred_np - target_np)
    sample_name = batch["sample_name"]

    sample_dir = plot_dir / sample_name
    plot_field(sample_dir / "target_temperature.png", pos, triangles, target_np, "Target temperature")
    plot_field(sample_dir / "predicted_temperature.png", pos, triangles, pred_np, "Predicted temperature")
    plot_field(sample_dir / "absolute_error.png", pos, triangles, abs_error, "Absolute error", cmap="magma")
    return sample_dir


def make_model(args, device, fun_dim=None):
    fun_dim = fun_dim if fun_dim is not None else args.fun_dim
    if fun_dim is None:
        fun_dim = 5
    return Model(
        space_dim=2,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        dropout=args.dropout,
        n_head=args.n_heads,
        Time_Input=False,
        mlp_ratio=args.mlp_ratio,
        fun_dim=fun_dim,
        out_dim=1,
        slice_num=args.slice_num,
        ref=args.ref,
        unified_pos=False,
    ).to(device)


def make_loader(dataset, shuffle=False):
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, collate_fn=heat2d_batch_size_one_collate)


def make_dataset(args, split, max_samples=None, sample_names=None):
    return Heat2DDataset(
        args.data_path,
        split=split,
        sample_names=sample_names,
        split_seed=args.split_seed,
        max_samples=max_samples,
        include_boundary_mask=args.include_boundary_mask,
    )


def relative_l2(pred, target):
    diff = torch.norm((pred - target).reshape(1, -1), p=2, dim=1)
    denom = torch.norm(target.reshape(1, -1), p=2, dim=1).clamp_min(1e-12)
    return (diff / denom).item()


def triangle_area_metrics(batch, pred, target):
    pos = batch["pos"][0].detach().cpu()
    triangles = batch["triangles"].detach().cpu().long()
    pred_values = pred[0, :, 0].detach().cpu()
    target_values = target[0, :, 0].detach().cpu()
    tri_pos = pos[triangles]
    edge_a = tri_pos[:, 1] - tri_pos[:, 0]
    edge_b = tri_pos[:, 2] - tri_pos[:, 0]
    areas = 0.5 * torch.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])
    total_area = areas.sum().clamp_min(1e-12)

    tri_pred = pred_values[triangles]
    tri_target = target_values[triangles]
    tri_sq_error = ((tri_pred - tri_target) ** 2).mean(dim=1)
    tri_sq_target = (tri_target**2).mean(dim=1)
    weighted_error = torch.sum(areas * tri_sq_error)
    weighted_target = torch.sum(areas * tri_sq_target).clamp_min(1e-12)
    return {
        "area_weighted_mse": (weighted_error / total_area).item(),
        "area_weighted_relative_l2": torch.sqrt(weighted_error / weighted_target).item(),
        "_area_total": total_area.item(),
        "_area_weighted_error_sum": weighted_error.item(),
        "_area_weighted_target_sum": weighted_target.item(),
    }


def summarize_rows(rows, prefix=""):
    if not rows:
        return {}
    rel_values = [row[f"{prefix}relative_l2"] for row in rows]
    mse_values = [row[f"{prefix}mse"] for row in rows]
    return {
        "mse": float(sum(mse_values) / len(mse_values)),
        "relative_l2": float(sum(rel_values) / len(rel_values)),
        "relative_l2_min": float(min(rel_values)),
        "relative_l2_mean": float(sum(rel_values) / len(rel_values)),
        "relative_l2_max": float(max(rel_values)),
    }


def evaluate(model, loader, stats, device, normalize=True, baseline=None):
    criterion = torch.nn.MSELoss()
    if model is not None:
        model.eval()
    results = []
    mse_total = 0.0
    rel_total = 0.0
    sq_error_total = 0.0
    sq_target_total = 0.0
    node_total = 0
    area_error_total = 0.0
    area_target_total = 0.0
    area_total = 0.0
    with torch.no_grad():
        for batch in loader:
            pos, fx, y = normalize_batch(batch, stats, device, normalize=normalize)
            if baseline == "zero":
                pred_physical = torch.zeros_like(decode_target(y, stats, normalize=normalize))
            elif baseline == "train_mean":
                target_physical = decode_target(y, stats, normalize=normalize)
                target_mean = stats["target_mean"].to(device).reshape(1, 1, 1)
                pred_physical = torch.zeros_like(target_physical) + target_mean
            else:
                pred = model(pos, fx=fx)
                pred_physical = decode_target(pred, stats, normalize=normalize)
            target_physical = decode_target(y, stats, normalize=normalize)
            mse = criterion(pred_physical, target_physical).item()
            rel = relative_l2(pred_physical, target_physical)
            abs_error = torch.abs(pred_physical - target_physical)
            area_metrics = triangle_area_metrics(batch, pred_physical.cpu(), target_physical.cpu())

            err_sq = torch.sum((pred_physical - target_physical) ** 2).item()
            target_sq = torch.sum(target_physical**2).item()
            sq_error_total += err_sq
            sq_target_total += target_sq
            node_total += int(target_physical.numel())

            area_total += area_metrics["_area_total"]
            area_error_total += area_metrics["_area_weighted_error_sum"]
            area_target_total += area_metrics["_area_weighted_target_sum"]

            mse_total += mse
            rel_total += rel
            manifest_entry = batch["manifest_entry"]
            results.append(
                {
                    "sample_name": batch["sample_name"],
                    "sample_id": batch["sample_id"],
                    "sample_path": batch["sample_path"],
                    "shape_type": batch["shape_type"],
                    "n_nodes": int(batch["pos"].shape[1]),
                    "n_elements": int(batch["triangles"].shape[0]),
                    "kappa_1": float(manifest_entry.get("kappa_1")),
                    "kappa_2": float(manifest_entry.get("kappa_2")),
                    "q_1": float(manifest_entry.get("q_1")),
                    "q_2": float(manifest_entry.get("q_2")),
                    "mse": mse,
                    "relative_l2": rel,
                    "max_absolute_error": abs_error.max().item(),
                    "mean_absolute_error": abs_error.mean().item(),
                    "area_weighted_mse": area_metrics["area_weighted_mse"],
                    "area_weighted_relative_l2": area_metrics["area_weighted_relative_l2"],
                    "batch": batch,
                    "pred": pred_physical.cpu(),
                    "target": target_physical.cpu(),
                }
            )

    count = max(len(results), 1)
    shape_rows = defaultdict(list)
    for row in results:
        shape_rows[row["shape_type"]].append(row)
    per_shape = {
        shape: {
            "mse": float(sum(row["mse"] for row in rows) / len(rows)),
            "relative_l2": float(sum(row["relative_l2"] for row in rows) / len(rows)),
            "area_weighted_mse": float(sum(row["area_weighted_mse"] for row in rows) / len(rows)),
            "area_weighted_relative_l2": float(
                sum(row["area_weighted_relative_l2"] for row in rows) / len(rows)
            ),
            "count": len(rows),
        }
        for shape, rows in sorted(shape_rows.items())
    }
    sample_rel = [row["relative_l2"] for row in results]
    area_rel_denom = max(area_target_total, 1e-12)
    return {
        "mse": mse_total / count,
        "relative_l2": rel_total / count,
        "global_nodal_mse": sq_error_total / max(node_total, 1),
        "global_nodal_relative_l2": (sq_error_total / max(sq_target_total, 1e-12)) ** 0.5,
        "area_weighted_mse": area_error_total / max(area_total, 1e-12),
        "area_weighted_relative_l2": (area_error_total / area_rel_denom) ** 0.5,
        "relative_l2_min": min(sample_rel) if sample_rel else None,
        "relative_l2_mean": sum(sample_rel) / len(sample_rel) if sample_rel else None,
        "relative_l2_max": max(sample_rel) if sample_rel else None,
        "per_shape": per_shape,
        "samples": results,
    }


def save_diagnostic_plots(eval_result, plot_dir, max_samples):
    saved_dirs = []
    for sample in eval_result["samples"][:max_samples]:
        sample_dir = save_overfit_plots(sample["batch"], sample["pred"], sample["target"], plot_dir)
        saved_dirs.append(str(sample_dir))
    return saved_dirs


def write_per_sample_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "sample_name",
        "sample_path",
        "shape_type",
        "n_nodes",
        "n_elements",
        "kappa_1",
        "kappa_2",
        "q_1",
        "q_2",
        "relative_l2",
        "mse",
        "area_weighted_relative_l2",
        "area_weighted_mse",
        "max_absolute_error",
        "mean_absolute_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def write_single_row_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def compact_eval_summary(eval_result):
    rows = eval_result["samples"]
    best = min(rows, key=lambda row: row["relative_l2"]) if rows else None
    worst = max(rows, key=lambda row: row["relative_l2"]) if rows else None
    return {
        "overall": {
            "mean_sample_mse": eval_result["mse"],
            "mean_sample_relative_l2": eval_result["relative_l2"],
            "global_nodal_mse": eval_result["global_nodal_mse"],
            "global_nodal_relative_l2": eval_result["global_nodal_relative_l2"],
            "area_weighted_mse": eval_result["area_weighted_mse"],
            "area_weighted_relative_l2": eval_result["area_weighted_relative_l2"],
            "relative_l2_min": eval_result["relative_l2_min"],
            "relative_l2_mean": eval_result["relative_l2_mean"],
            "relative_l2_max": eval_result["relative_l2_max"],
        },
        "per_shape": eval_result["per_shape"],
        "best_sample": None
        if best is None
        else {
            "sample_name": best["sample_name"],
            "shape_type": best["shape_type"],
            "relative_l2": best["relative_l2"],
            "mse": best["mse"],
        },
        "worst_sample": None
        if worst is None
        else {
            "sample_name": worst["sample_name"],
            "shape_type": worst["shape_type"],
            "relative_l2": worst["relative_l2"],
            "mse": worst["mse"],
        },
    }


def save_checkpoint(path, model, optimizer, args, stats, epoch, train_loss, eval_metrics, best_rel_l2):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "normalization_stats": stats,
            "epoch": epoch,
            "train_loss": train_loss,
            "eval_metrics": eval_metrics,
            "best_validation_relative_l2": best_rel_l2,
        },
        path,
    )


def save_learning_curves(path, train_losses, val_relative_l2):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    epochs = list(range(1, len(train_losses) + 1))
    ax1.plot(epochs, train_losses, label="train loss", color="tab:blue")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train normalized MSE", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    val_epochs = [epoch for epoch, value in enumerate(val_relative_l2, start=1) if value is not None]
    val_values = [value for value in val_relative_l2 if value is not None]
    ax2.plot(val_epochs, val_values, label="val relative L2", color="tab:orange")
    ax2.set_ylabel("validation relative L2", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_smoke(args):
    dataset = make_dataset(args, args.split, max_samples=args.max_samples)
    all_finite = True
    y_min = None
    y_max = None
    first = None

    for item in dataset:
        if first is None:
            first = item
        arrays = [item["pos"], item["node_features"], item["target"]]
        all_finite = all_finite and all(bool(torch.isfinite(array).all()) for array in arrays)
        sample_min = float(item["target"].min())
        sample_max = float(item["target"].max())
        y_min = sample_min if y_min is None else min(y_min, sample_min)
        y_max = sample_max if y_max is None else max(y_max, sample_max)

    print(f"number of samples: {len(dataset)}")
    print(f"x shape: {tuple(first['pos'].shape)}")
    print(f"fx shape: {tuple(first['node_features'].shape)}")
    print(f"y shape: {tuple(first['target'].shape)}")
    print(f"min/max of y: {y_min:.8e} {y_max:.8e}")
    print(f"all arrays finite: {all_finite}")
    print(f"input features: {dataset.input_feature_names}")
    print("batch_size limitation: variable-size meshes currently use batch_size=1")


def run_overfit(args):
    torch.manual_seed(args.seed)

    os.makedirs(args.plot_dir, exist_ok=True)
    device = get_device(args.device)
    normalize = not args.no_normalize

    stats_dataset = make_dataset(args, args.split, max_samples=args.max_samples)
    sample_name = args.sample_name or stats_dataset.samples[0]["sample_name"]
    dataset = make_dataset(args, args.split, sample_names=[sample_name])
    loader = make_loader(dataset, shuffle=False)
    batch = next(iter(loader))

    stats = compute_stats(stats_dataset)
    pos, fx, y = normalize_batch(batch, stats, device, normalize=normalize)

    model = make_model(args, device, fun_dim=dataset.num_input_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()

    model.eval()
    with torch.no_grad():
        initial_pred = model(pos, fx=fx)
        initial_loss = criterion(initial_pred, y).item()

    model.train()
    final_loss = initial_loss
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        pred = model(pos, fx=fx)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if args.print_every > 0 and (step == 1 or step % args.print_every == 0 or step == args.steps):
            print(f"step {step:05d} loss {final_loss:.8e}")

    model.eval()
    with torch.no_grad():
        pred = model(pos, fx=fx)
        final_loss = criterion(pred, y).item()
        pred_plot = decode_target(pred, stats, normalize=normalize)
        target_plot = decode_target(y, stats, normalize=normalize)
        physical_mse = criterion(pred_plot, target_plot).item()

    plot_dir = save_overfit_plots(batch, pred_plot.cpu(), target_plot.cpu(), args.plot_dir)

    print(f"sample_name: {sample_name}")
    print(f"device: {device}")
    print(f"normalization: {'on' if normalize else 'off'}")
    print(f"initial_loss: {initial_loss:.8e}")
    print(f"final_loss: {final_loss:.8e}")
    print(f"physical_mse: {physical_mse:.8e}")
    print(f"plots: {plot_dir}")


def run_train(args):
    if args.batch_size != 1:
        raise ValueError("Heat2D currently supports --batch-size 1 only.")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1.")

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    normalize = not args.no_normalize
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = make_dataset(args, args.train_split, max_samples=args.max_train_samples)
    val_dataset = make_dataset(args, args.val_split, max_samples=args.max_val_samples)
    test_dataset = make_dataset(args, args.test_split, max_samples=args.max_test_samples)
    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False)
    test_loader = make_loader(test_dataset, shuffle=False)

    stats = compute_stats(train_dataset)
    args.fun_dim = train_dataset.num_input_features
    model = make_model(args, device, fun_dim=train_dataset.num_input_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()

    config_path = output_dir / "config.json"
    stats_path = output_dir / "normalization_stats.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats_to_jsonable(stats), f, indent=2)

    best_val_rel_l2 = float("inf")
    final_train_loss = None
    final_val_eval = None
    train_losses = []
    val_relative_l2 = []
    optimizer.zero_grad()

    print(f"train_samples: {len(train_dataset)}")
    print(f"val_samples: {len(val_dataset)}")
    print(f"test_samples: {len(test_dataset)}")
    print(f"device: {device}")
    print(f"normalization: {'on' if normalize else 'off'}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, start=1):
            pos, fx, y = normalize_batch(batch, stats, device, normalize=normalize)
            pred = model(pos, fx=fx)
            loss = criterion(pred, y)
            (loss / args.grad_accum_steps).backward()
            train_loss_total += loss.item()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        final_train_loss = train_loss_total / len(train_loader)
        train_losses.append(final_train_loss)

        should_eval = epoch == args.epochs or args.eval_every > 0 and epoch % args.eval_every == 0
        if should_eval:
            final_val_eval = evaluate(model, val_loader, stats, device, normalize=normalize)
            val_rel_l2_score = final_val_eval["relative_l2"]
            val_mse = final_val_eval["mse"]
            val_relative_l2.append(val_rel_l2_score)
            if val_rel_l2_score < best_val_rel_l2:
                best_val_rel_l2 = val_rel_l2_score
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    args,
                    stats,
                    epoch,
                    final_train_loss,
                    {"split": args.val_split, "mse": val_mse, "relative_l2": val_rel_l2_score},
                    best_val_rel_l2,
                )
            print(
                f"epoch {epoch:04d} train_loss {final_train_loss:.8e} "
                f"val_mse {val_mse:.8e} val_relative_l2 {val_rel_l2_score:.8e} "
                f"best_val_relative_l2 {best_val_rel_l2:.8e}"
            )
        else:
            val_relative_l2.append(None)
            print(f"epoch {epoch:04d} train_loss {final_train_loss:.8e}")

    save_checkpoint(
        checkpoint_dir / "last.pt",
        model,
        optimizer,
        args,
        stats,
        args.epochs,
        final_train_loss,
        {"split": args.val_split, "mse": final_val_eval["mse"], "relative_l2": final_val_eval["relative_l2"]},
        best_val_rel_l2,
    )

    best_path = checkpoint_dir / "best.pt"
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_eval = evaluate(model, test_loader, stats, device, normalize=normalize)
    saved_plot_dirs = save_diagnostic_plots(test_eval, plot_dir, args.plot_samples)
    learning_curves = {
        "train_loss": train_losses,
        "val_relative_l2": val_relative_l2,
    }
    with (output_dir / "learning_curves.json").open("w", encoding="utf-8") as f:
        json.dump(learning_curves, f, indent=2)
    save_learning_curves(output_dir / "learning_curves.png", train_losses, val_relative_l2)

    metrics = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "final_train_loss": final_train_loss,
        "best_validation_relative_l2": best_val_rel_l2,
        "best_checkpoint_epoch": int(best_checkpoint["epoch"]),
        "test_mse_from_best_validation": test_eval["mse"],
        "test_relative_l2_from_best_validation": test_eval["relative_l2"],
        "checkpoints": {
            "best": str(checkpoint_dir / "best.pt"),
            "last": str(checkpoint_dir / "last.pt"),
        },
        "plot_dirs": saved_plot_dirs,
        "per_sample_test_metrics": [
            {
                "sample_name": sample["sample_name"],
                "mse": sample["mse"],
                "relative_l2": sample["relative_l2"],
            }
            for sample in test_eval["samples"]
        ],
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    summary = {
        "run_type": "train",
        "output_dir": str(output_dir),
        "data_path": args.data_path,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "model": {
            "n_hidden": args.n_hidden,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "slice_num": args.slice_num,
            "fun_dim": args.fun_dim,
        },
        "epochs": args.epochs,
        "best_checkpoint_epoch": int(best_checkpoint["epoch"]),
        "final_train_loss": final_train_loss,
        "best_validation_relative_l2": best_val_rel_l2,
        "test_mse_from_best_validation": test_eval["mse"],
        "test_relative_l2_from_best_validation": test_eval["relative_l2"],
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "last_checkpoint": str(checkpoint_dir / "last.pt"),
        "plots": str(plot_dir),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    csv_summary = {key: value for key, value in summary.items() if key != "model"} | summary["model"]
    write_single_row_csv(output_dir / "metrics.csv", csv_summary)
    write_per_sample_csv(output_dir / "per_sample_metrics.csv", test_eval["samples"])

    print(f"final_train_loss: {final_train_loss:.8e}")
    print(f"best_validation_relative_l2: {best_val_rel_l2:.8e}")
    print(f"test_mse_from_best_validation: {test_eval['mse']:.8e}")
    print(f"test_relative_l2_from_best_validation: {test_eval['relative_l2']:.8e}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"plot_dir: {plot_dir}")
    print(f"metrics: {output_dir / 'metrics.json'}")
    print(f"run_summary: {output_dir / 'run_summary.json'}")


def run_eval(args):
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --mode eval.")

    device = get_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    for key in ["n_hidden", "n_layers", "n_heads", "mlp_ratio", "dropout", "slice_num", "ref"]:
        if key in saved_args:
            setattr(args, key, saved_args[key])
    if "include_boundary_mask" in saved_args:
        args.include_boundary_mask = bool(saved_args["include_boundary_mask"])
    if "split_seed" in saved_args:
        args.split_seed = int(saved_args["split_seed"])
    normalize = not bool(saved_args.get("no_normalize", args.no_normalize))

    output_dir = Path(args.eval_output_dir) if args.eval_output_dir else checkpoint_path.parent.parent / "diagnostics"
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = make_dataset(args, args.test_split, max_samples=args.max_test_samples)
    test_loader = make_loader(test_dataset, shuffle=False)
    stats = load_stats_from_checkpoint(checkpoint)

    if args.fun_dim is None:
        args.fun_dim = int(saved_args.get("fun_dim", test_dataset.num_input_features))
    model = make_model(args, device, fun_dim=args.fun_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model_eval = evaluate(model, test_loader, stats, device, normalize=normalize)
    zero_eval = evaluate(None, test_loader, stats, device, normalize=normalize, baseline="zero")
    mean_eval = evaluate(None, test_loader, stats, device, normalize=normalize, baseline="train_mean")

    saved_plot_dirs = save_diagnostic_plots(model_eval, plot_dir, args.plot_samples)
    write_per_sample_csv(output_dir / "per_sample_metrics.csv", model_eval["samples"])

    diagnostics = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "test_split": args.test_split,
        "normalize": normalize,
        "model": compact_eval_summary(model_eval),
        "baselines": {
            "zero_prediction": compact_eval_summary(zero_eval)["overall"],
            "train_mean_prediction": compact_eval_summary(mean_eval)["overall"],
        },
        "plot_dirs": saved_plot_dirs,
        "per_sample_csv": str(output_dir / "per_sample_metrics.csv"),
    }
    with (output_dir / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    summary = {
        "run_type": "eval",
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": diagnostics["checkpoint_epoch"],
        "test_split": args.test_split,
        "test_samples": len(test_dataset),
        "overall_mse": diagnostics["model"]["overall"]["mean_sample_mse"],
        "overall_relative_l2": diagnostics["model"]["overall"]["mean_sample_relative_l2"],
        "global_nodal_mse": diagnostics["model"]["overall"]["global_nodal_mse"],
        "global_nodal_relative_l2": diagnostics["model"]["overall"]["global_nodal_relative_l2"],
        "area_weighted_mse": diagnostics["model"]["overall"]["area_weighted_mse"],
        "area_weighted_relative_l2": diagnostics["model"]["overall"]["area_weighted_relative_l2"],
        "zero_prediction_relative_l2": diagnostics["baselines"]["zero_prediction"]["mean_sample_relative_l2"],
        "train_mean_prediction_relative_l2": diagnostics["baselines"]["train_mean_prediction"][
            "mean_sample_relative_l2"
        ],
        "best_sample": diagnostics["model"]["best_sample"]["sample_name"],
        "best_sample_relative_l2": diagnostics["model"]["best_sample"]["relative_l2"],
        "worst_sample": diagnostics["model"]["worst_sample"]["sample_name"],
        "worst_sample_relative_l2": diagnostics["model"]["worst_sample"]["relative_l2"],
        "plots": str(plot_dir),
        "per_sample_csv": str(output_dir / "per_sample_metrics.csv"),
    }
    for shape, metrics in diagnostics["model"]["per_shape"].items():
        summary[f"{shape}_relative_l2"] = metrics["relative_l2"]
        summary[f"{shape}_mse"] = metrics["mse"]
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_single_row_csv(output_dir / "metrics.csv", summary)

    overall = diagnostics["model"]["overall"]
    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint_epoch: {diagnostics['checkpoint_epoch']}")
    print(f"test_samples: {len(test_dataset)}")
    print(f"overall_mse: {overall['mean_sample_mse']:.8e}")
    print(f"overall_relative_l2: {overall['mean_sample_relative_l2']:.8e}")
    print(f"global_nodal_mse: {overall['global_nodal_mse']:.8e}")
    print(f"global_nodal_relative_l2: {overall['global_nodal_relative_l2']:.8e}")
    print(f"area_weighted_mse: {overall['area_weighted_mse']:.8e}")
    print(f"area_weighted_relative_l2: {overall['area_weighted_relative_l2']:.8e}")
    for shape, metrics in diagnostics["model"]["per_shape"].items():
        print(
            f"shape {shape}: mse {metrics['mse']:.8e} "
            f"relative_l2 {metrics['relative_l2']:.8e} "
            f"area_relative_l2 {metrics['area_weighted_relative_l2']:.8e}"
        )
    print(
        "baseline_zero_relative_l2: "
        f"{diagnostics['baselines']['zero_prediction']['mean_sample_relative_l2']:.8e}"
    )
    print(
        "baseline_train_mean_relative_l2: "
        f"{diagnostics['baselines']['train_mean_prediction']['mean_sample_relative_l2']:.8e}"
    )
    print(f"best_sample: {diagnostics['model']['best_sample']}")
    print(f"worst_sample: {diagnostics['model']['worst_sample']}")
    print(f"diagnostics: {output_dir / 'diagnostics.json'}")
    print(f"run_summary: {output_dir / 'run_summary.json'}")
    print(f"per_sample_csv: {output_dir / 'per_sample_metrics.csv'}")
    print(f"plot_dir: {plot_dir}")


def main():
    args = build_parser().parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    elif args.mode == "overfit":
        run_overfit(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
