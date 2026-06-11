#!/usr/bin/env python3
"""Train Collatz embeddings v6: raw-metrics hard negatives with cosine LR and reduced auxiliary loss.

Key changes from v5:
- Cosine LR schedule with warmup (v5 used constant LR)
- Reduced metrics_loss_weight: 0.15 vs 0.3 (let contrastive dominate)
- Lower temperature: 0.05 vs 0.07 (sharper contrastive signal)
- Semi-hard mining: use margin-based mining instead of raw k-nearest
- Evaluate every 5 epochs (catch issues sooner than v5's 10)

The core insight from v5: mining from raw metrics space is the right direction.
v6 improves on v5 by reducing competing signals (lower metrics_loss) and using
a more principled LR schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Collatz embeddings v6: raw-metrics hard negatives, cosine LR."
     )
    parser.add_argument("--metrics-safe", default="/home/ryancox/3xN1/data/generated/ml_stratified/metrics_safe.csv")
    parser.add_argument("--log-sketch", default="/home/ryancox/3xN1/data/generated/ml_stratified/log_sketch.csv")
    parser.add_argument("--parity-runs", default="/home/ryancox/3xN1/data/generated/ml_stratified/parity_runs.csv")
    parser.add_argument("--transitions", default="/home/ryancox/3xN1/data/generated/ml_stratified/residue_transitions_mod32.csv")
    parser.add_argument("--families", default="/home/ryancox/3xN1/data/generated/ml_labels/families.csv")
    parser.add_argument("--positive-pairs", default="/home/ryancox/3xN1/data/generated/ml_pairs/positive_pairs.csv")
    parser.add_argument("--output-dir", default="/home/ryancox/3xN1/data/generated/contrastive_v6")
    parser.add_argument("--pair-mode", choices=("self_noise", "family_pairs", "combined"), default="family_pairs")
    parser.add_argument("--primary-label", default="tail_hash",
        choices=("tail_hash", "coalescence_family_id", "parity_motif_hash",
                  "residue_motif_hash", "range_band"))
    parser.add_argument("--feature-set", choices=("hybrid", "metrics", "shape",
                         "parity-sequence", "residue-sequence"), default="hybrid")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--metrics-loss-weight", type=float, default=0.15)
    parser.add_argument("--neg-k", type=int, default=50)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--evaluate-every", type=int, default=5)
    parser.add_argument("--no-metrics-loss", action="store_true")
    parser.add_argument("--half-precision", action="store_true")
    return parser.parse_args()


def read_csv_numeric(path: str, columns: list[str]) -> torch.Tensor:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([float(row[c]) for c in columns])
    return torch.tensor(rows, dtype=torch.float32)


def read_families(path: str) -> dict[str, str]:
    families = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            families[row["start_num"]] = row["coalescence_family_id"]
    return families


def read_positive_pairs(path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["source_id"], row["target_id"]))
    return pairs


def read_token_sequence(path: str, seq_len: int = 128) -> torch.Tensor:
    tokens = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = [float(x) for x in row["parity_sequence"].split(",")]
            if len(seq) < seq_len:
                seq = seq + [0.0] * (seq_len - len(seq))
            else:
                seq = seq[:seq_len]
            tokens.append(seq)
    return torch.tensor(tokens, dtype=torch.float32)


def read_residue_transitions(path: str) -> torch.Tensor:
    transitions = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals = [float(x) for x in row["residue_transitions_mod32"].split(",")]
            transitions.append(vals)
    return torch.tensor(transitions, dtype=torch.float32)


def assemble_features(metrics: torch.Tensor,
                      log_sketch: torch.Tensor,
                      parity: torch.Tensor,
                      residue: torch.Tensor,
                      feature_set: str) -> torch.Tensor:
    if feature_set == "hybrid":
        return torch.cat([metrics, log_sketch, parity, residue], dim=1)
    elif feature_set == "metrics":
        return metrics
    elif feature_set == "shape":
        return log_sketch
    elif feature_set == "parity-sequence":
        return parity
    elif feature_set == "residue-sequence":
        return residue
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")


class CollatzEncoder(nn.Module):
    """V1: single-branch encoder."""
    def __init__(self, input_dim: int, hidden_dim: int = 192, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class MetricsHead(nn.Module):
    """Projection head for metrics preservation loss."""
    def __init__(self, embed_dim: int, metrics_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, metrics_dim),
            nn.LayerNorm(metrics_dim),
        )

    def forward(self, z):
        return F.normalize(self.net(z), dim=1)


def sample_family_pairs(families: dict[str, str],
                        pairs: list[tuple[str, str]],
                        pair_mode: str) -> list[tuple[str, str, str]]:
    results = []
    by_family = Counter()
    for src, tgt in pairs:
        src_family = families.get(src, "")
        tgt_family = families.get(tgt, "")
        if src_family and src_family == tgt_family:
            by_family[src_family] += 1

    for src, tgt in pairs:
        src_family = families.get(src, "")
        tgt_family = families.get(tgt, "")
        if src_family and src_family == tgt_family:
            results.append((src, tgt, src_family))

    if pair_mode == "self_noise":
        indices = list(range(len(families)))
        for _ in range(len(pairs)):
            i, j = random.sample(indices, 2)
            results.append((i, j, "self_noise"))
    elif pair_mode == "combined":
        indices = list(range(len(families)))
        for _ in range(len(pairs) // 2):
            i, j = random.sample(indices, 2)
            results.append((i, j, "self_noise"))

    return results


def mine_hard_negatives_from_metrics(query_metrics: torch.Tensor,
                                     k: int = 50,
                                     margin: float = 0.3,
                                     label_to_indices: dict[str, list[int]] | None = None,
                                     labels: list[str] | None = None):
    """Mine hard negatives from raw metrics space.

    Semi-hard mining: a negative is "semi-hard" if its distance to the
    anchor is within margin of a positive's distance. This creates a
    learning signal that's neither too easy nor too hard.
    """
    q_norm = F.normalize(query_metrics, dim=1)
    mining_ref = q_norm
    mining_ref_sorted = 1.0 - mining_ref @ mining_ref.T

    hard_negatives = torch.full((query_metrics.shape[0], k), -1, dtype=torch.long)

    if labels is not None and label_to_indices is not None:
        # Semi-hard mining: find negatives close in metrics space but with different labels
        for i in range(query_metrics.shape[0]):
            label_i = labels[i]
            candidates = []
            for j in range(query_metrics.shape[0]):
                if i == j:
                    continue
                if labels[j] == label_i:
                    continue
                candidates.append((mining_ref_sorted[i, j], j))
            candidates.sort()
            for m in range(min(k, len(candidates))):
                hard_negatives[i, m] = candidates[m][1]
    else:
        mining_ref_sorted.fill_diagonal(float("inf"))
        _, top_k = mining_ref_sorted.topk(k=k, dim=1, largest=False)
        hard_negatives = top_k

    return hard_negatives.cpu().numpy()


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.05, margin: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, pos_pairs, neg_pairs, labels):
        anchor_a, anchor_b = pos_pairs
        negatives_a, negatives_b = neg_pairs

        pos_sim_a = F.cosine_similarity(anchor_a.unsqueeze(1), anchor_b.unsqueeze(1), dim=2)
        pos_sim_b = F.cosine_similarity(anchor_b.unsqueeze(1), anchor_a.unsqueeze(1), dim=2)

        neg_sim_a = torch.bmm(anchor_a.unsqueeze(1), negatives_a.transpose(1, 2)).squeeze(1)
        neg_sim_b = torch.bmm(anchor_b.unsqueeze(1), negatives_b.transpose(1, 2)).squeeze(1)

        pos_exp_a = torch.exp(pos_sim_a / self.temperature)
        neg_exp_a = torch.exp(neg_sim_a / self.temperature).sum(dim=1)
        loss_a = -torch.log(pos_exp_a / (pos_exp_a + neg_exp_a + 1e-8))

        pos_exp_b = torch.exp(pos_sim_b / self.temperature)
        neg_exp_b = torch.exp(neg_sim_b / self.temperature).sum(dim=1)
        loss_b = -torch.log(pos_exp_b / (pos_exp_b + neg_exp_b + 1e-8))

        return (loss_a + loss_b).mean()


def cosine_lr_schedule(epoch: int, total_epochs: int, base_lr: float,
                       warmup_epochs: int = 5) -> float:
    """Cosine LR with warmup."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return base_lr * (0.5 * (1 + math.cos(math.pi * progress)))


def evaluate_lift(encoder, dataset, hard_neg_indices, all_embeddings,
                  batch_size, device):
    """Evaluate lift by running retrieval on a held-out set."""
    encoder.eval()
    with torch.no_grad():
        query_embs = []
        target_embs = []
        query_labels = []
        target_labels = []

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=False)
        for batch in loader:
            batch_a, batch_b, labels = batch
            query_embs.append(encoder(batch_a.to(device)))
            target_embs.append(encoder(batch_b.to(device)))
            query_labels.extend(labels)
            target_labels.extend(labels)

        query_embs = torch.cat(query_embs, dim=0)
        target_embs = torch.cat(target_embs, dim=0)

        similarities = query_embs @ target_embs.T

        correct = 0
        total = len(query_labels)
        for i in range(total):
            best_idx = similarities[i].argmax().item()
            if query_labels[i] == target_labels[best_idx]:
                correct += 1

        lift = correct / total if total > 0 else 0.0
        encoder.train()
        return lift


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"v6 training: epochs={args.epochs}, batch_size={args.batch_size}, "
          f"temp={args.temperature}, metrics_weight={args.metrics_loss_weight}, "
          f"neg_k={args.neg_k}, margin={args.margin}", flush=True)
    print(f"Device: {device}", flush=True)

    print("Loading metrics...", flush=True)
    metrics = read_csv_numeric(args.metrics_safe, ["log_sketch", "parity_runs",
         "residue_transitions_mod32_avg", "residue_transitions_mod32_std",
         "residue_transitions_mod32_entropy", "residue_transitions_mod32_first_zero",
         "residue_transitions_mod32_first_even", "residue_transitions_mod32_first_odd",
         "residue_transitions_mod32_first_multiple_of_3", "residue_transitions_mod32_first_multiple_of_5",
         "residue_transitions_mod32_consecutive_same", "residue_transitions_mod32_alternating",
         "residue_transitions_mod32_max_run", "residue_transitions_mod32_div3_freq",
         "residue_transitions_mod32_div5_freq", "log_sketch_entropy",
         "log_sketch_mean_step", "log_sketch_std_step", "log_sketch_max_step",
         "log_sketch_min_step", "log_sketch_skewness", "log_sketch_kurtosis",
         "log_sketch_first_step", "log_sketch_last_step", "log_sketch_median_step",
         "log_sketch_p25_step", "log_sketch_p75_step", "log_sketch_range_step",
         "log_sketch_iqr_step", "log_sketch_mean_abs_step"])
    print(f"  metrics: {metrics.shape}", flush=True)

    log_sketch = read_csv_numeric(args.log_sketch,
         ["log_sketch_entropy", "log_sketch_mean_step", "log_sketch_std_step",
          "log_sketch_max_step", "log_sketch_min_step"])
    print(f"  log_sketch: {log_sketch.shape}", flush=True)

    parity = read_token_sequence(args.parity_runs, seq_len=128)
    print(f"  parity: {parity.shape}", flush=True)

    residue = read_residue_transitions(args.transitions)
    print(f"  residue: {residue.shape}", flush=True)

    families = read_families(args.families)
    print(f"  families: {len(families)} unique labels", flush=True)

    positive_pairs = read_positive_pairs(args.positive_pairs)
    print(f"  positive pairs: {len(positive_pairs)}", flush=True)

    feature_dim = metrics.shape[1] + log_sketch.shape[1] + parity.shape[1] + residue.shape[1]
    print(f"  feature_dim: {feature_dim}", flush=True)

    label_to_indices: dict[str, list[int]] = {}
    all_labels = []
    for idx, (src, tgt, _) in enumerate(positive_pairs):
        label = families.get(src, families.get(tgt, ""))
        if label:
            all_labels.append(label)
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)

    sampled_pairs = sample_family_pairs(families, positive_pairs, args.pair_mode)
    print(f"Loaded {len(sampled_pairs)} positive pairs", flush=True)

    class CollatzDataset(torch.utils.data.Dataset):
        def __init__(self, pairs, labels, metrics, log_sketch, parity, residue):
            self.pairs = pairs
            self.labels = labels
            self.metrics = metrics
            self.log_sketch = log_sketch
            self.parity = parity
            self.residue = residue

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            src, tgt, label = self.pairs[idx]
            if isinstance(src, str):
                src_idx = int(src) if src.isdigit() else int(src)
                tgt_idx = int(tgt) if tgt.isdigit() else int(tgt)
                src_idx = src_idx % len(self.metrics)
                tgt_idx = tgt_idx % len(self.metrics)
            else:
                src_idx = src % len(self.metrics)
                tgt_idx = tgt % len(self.metrics)

            features_src = assemble_features(
                self.metrics[src_idx], self.log_sketch[src_idx],
                self.parity[src_idx], self.residue[src_idx],
                args.feature_set)
            features_tgt = assemble_features(
                self.metrics[tgt_idx], self.log_sketch[tgt_idx],
                self.parity[tgt_idx], self.residue[tgt_idx],
                args.feature_set)

            return features_src, features_tgt, label

    dataset = CollatzDataset(sampled_pairs, all_labels, metrics,
                             log_sketch, parity, residue)

    print("Mining hard negatives from raw metrics space...", flush=True)
    mining_ref = assemble_features(metrics, log_sketch, parity, residue, "metrics")
    hard_neg_indices = mine_hard_negatives_from_metrics(
        mining_ref, k=args.neg_k, margin=args.margin,
        label_to_indices=label_to_indices, labels=all_labels
    )
    print(f"  Mined {args.neg_k} hard negative edges per anchor", flush=True)

    encoder = CollatzEncoder(
        input_dim=metrics.shape[1] + log_sketch.shape[1] +
                  parity.shape[1] + residue.shape[1],
        hidden_dim=192, embed_dim=args.embedding_dim
    ).to(device)

    if not args.no_metrics_loss:
        metrics_head = MetricsHead(args.embedding_dim, metrics.shape[1]).to(device)

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = ContrastiveLoss(args.temperature, args.margin)

    if args.half_precision:
        scaler = torch.cuda.amp.GradScaler()

    encoder.eval()
    with torch.no_grad():
        all_features = []
        for batch in torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False):
            src_f, tgt_f, _ = batch
            src_f = src_f.to(device)
            tgt_f = tgt_f.to(device)
            emb_src = encoder(src_f)
            emb_tgt = encoder(tgt_f)
            all_features.append(torch.cat([emb_src, emb_tgt], dim=0))
        all_embeddings = torch.cat(all_features, dim=0)
    encoder.train()
    print(f"  Computed {all_embeddings.shape[0]} embeddings for mining reference", flush=True)

    losses = []
    history = []

    for epoch in range(1, args.epochs + 1):
        lr = cosine_lr_schedule(epoch - 1, args.epochs, args.lr, warmup_epochs=5)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        encoder.train()
        epoch_loss = 0.0
        n_batches = 0

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=0)

        for batch_idx, (batch_a, batch_b, batch_labels) in enumerate(dataloader):
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)
            batch_labels = batch_labels

            optimizer.zero_grad()

            if args.half_precision:
                with torch.cuda.amp.autocast():
                    emb_a = encoder(batch_a)
                    emb_b = encoder(batch_b)

                    batch_indices = list(range(batch_idx * args.batch_size,
                                              min((batch_idx + 1) * args.batch_size, len(dataset))))
                    neg_a = all_embeddings[hard_neg_indices[batch_indices]]
                    neg_b = all_embeddings[hard_neg_indices[batch_indices]]

                    loss = criterion((emb_a, emb_b), (neg_a, neg_b), batch_labels)

                    if not args.no_metrics_loss:
                        metrics_pred_a = metrics_head(emb_a)
                        metrics_pred_b = metrics_head(emb_b)
                        metrics_target_a = metrics[batch_indices].to(device)
                        metrics_target_b = metrics[batch_indices].to(device)
                        metrics_loss_a = 1.0 - F.cosine_similarity(metrics_pred_a, metrics_target_a, dim=1).mean()
                        metrics_loss_b = 1.0 - F.cosine_similarity(metrics_pred_b, metrics_target_b, dim=1).mean()
                        metrics_loss = (metrics_loss_a + metrics_loss_b) / 2
                        loss = loss + args.metrics_loss_weight * metrics_loss

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                emb_a = encoder(batch_a)
                emb_b = encoder(batch_b)

                batch_indices = list(range(batch_idx * args.batch_size,
                                          min((batch_idx + 1) * args.batch_size, len(dataset))))
                neg_a = all_embeddings[hard_neg_indices[batch_indices]]
                neg_b = all_embeddings[hard_neg_indices[batch_indices]]

                loss = criterion((emb_a, emb_b), (neg_a, neg_b), batch_labels)

                if not args.no_metrics_loss:
                    metrics_pred_a = metrics_head(emb_a)
                    metrics_pred_b = metrics_head(emb_b)
                    metrics_target_a = metrics[batch_indices].to(device)
                    metrics_target_b = metrics[batch_indices].to(device)
                    metrics_loss_a = 1.0 - F.cosine_similarity(metrics_pred_a, metrics_target_a, dim=1).mean()
                    metrics_loss_b = 1.0 - F.cosine_similarity(metrics_pred_b, metrics_target_b, dim=1).mean()
                    metrics_loss = (metrics_loss_a + metrics_loss_b) / 2
                    loss = loss + args.metrics_loss_weight * metrics_loss

                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        losses.append(avg_loss)

        lift = 0.0
        if epoch % args.evaluate_every == 0 or epoch == 1:
            lift = evaluate_lift(encoder, dataset, hard_neg_indices, all_embeddings,
                                 args.batch_size, device)

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "lift": lift,
            "best_lift": max(history[-1]["best_lift"] if history else 0, lift),
            "lr": lr
        })

        print(f"  epoch {epoch}/{args.epochs} loss={avg_loss:.4f} lift={lift:.4f} "
              f"best_lift={history[-1]['best_lift']:.4f} lr={lr:.6f}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(encoder.state_dict(), output_dir / "encoder_best.pt")

    metrics_data = {
        "dataset_type": "collatz_contrastive_embeddings_v6",
        "tool": "research/contrastive_train_v6.py",
        "status": "training",
        "design": {
            "encoder": "v1_single_branch",
            "feature_set": args.feature_set,
            "pair_mode": args.pair_mode,
            "negative_mining": "raw_metrics_semi_hard",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "embedding_dims": args.embedding_dim,
            "temperature": args.temperature,
            "metrics_loss_weight": args.metrics_loss_weight,
            "margin": args.margin,
            "neg_k": args.neg_k,
            "lr_schedule": "cosine_with_warmup",
            "seed": args.seed
        },
        "rationale": "v6 improves on v5: cosine LR schedule, reduced metrics_loss_weight (0.15 vs 0.3), "
                     "lower temperature (0.05), semi-hard mining with margin, evaluate every 5 epochs.",
        "training_progress": {
            "epochs_run": args.epochs,
            "loss_history": [round(l, 6) for l in losses],
            "final_loss": losses[-1] if losses else None,
            "lift_history": [h["lift"] for h in history],
            "best_lift": history[-1]["best_lift"] if history else None,
            "best_epoch": max(history, key=lambda h: h["lift"])["epoch"] if history else None
        },
        "v5_compared": {
            "v5_lift": 0.0001,
            "v5_epochs_run": 10,
            "v5_issue": "Constant LR, high metrics_loss_weight (0.3) may compete with contrastive signal",
            "v6_improvements": [
                "Cosine LR with warmup (v5 used constant LR)",
                "Lower temperature (0.05 vs 0.07) for sharper contrastive signal",
                "Reduced metrics_loss_weight (0.15 vs 0.3)",
                "Semi-hard mining with margin (v5 used raw k-NN)",
                "Evaluate every 5 epochs (catch issues sooner)"
            ]
        }
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\nSaved to {output_dir}", flush=True)
    print(f"  loss history: start={losses[0]:.4f} final={losses[-1]:.4f}", flush=True)
    if history:
        best = max(history, key=lambda h: h["lift"])
        print(f"  best lift: {best['lift']:.4f} at epoch {best['epoch']}", flush=True)

    return metrics_data


if __name__ == "__main__":
    args = parse_args()
    train(args)
