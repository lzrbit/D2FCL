"""
Generator for CIFAR-100 federated continual learning split.

Two modes (--mode):
  ltp    (default) — LTP setting: each client sees all 100 classes in a different random
                     task ordering. 10 clients × 10 tasks × 10 classes/task.
                     Output: CIFAR100_split_cn10_tn10_cet10_cs1_s2571.pkl

  random           — Random sampling: each client independently samples 20 classes per task
                     (classes may overlap across tasks). 10 clients × 4 tasks × 20 classes/task.
                     Output: CIFAR100_split_cn10_tn4_cet20_s2571.pkl

Usage:
  python scripts/generate_cifar100_split.py            # ltp mode (default)
  python scripts/generate_cifar100_split.py --mode random
"""

import argparse
import os
import pickle
import numpy as np
from torchvision import datasets

# ── Shared constants ─────────────────────────────────────────────────────────
N_CLIENTS  = 10
N_CLASSES  = 100
SEED       = 2571
DATA_DIR   = "datasets"
SPLIT_DIR  = os.path.join(DATA_DIR, "split_files")


def load_cifar100():
    """Load CIFAR-100 and build per-class index maps."""
    print("Loading CIFAR-100 …")
    cifar_train = datasets.CIFAR100(DATA_DIR, train=True,  download=False)
    cifar_test  = datasets.CIFAR100(DATA_DIR, train=False, download=False)

    train_labels = np.array(cifar_train.targets)
    test_labels  = np.array(cifar_test.targets)

    train_class_inds = {c: np.where(train_labels == c)[0].tolist() for c in range(N_CLASSES)}
    test_class_inds  = {c: np.where(test_labels  == c)[0].tolist() for c in range(N_CLASSES)}

    print(f"Train: {len(train_labels)} samples, Test: {len(test_labels)} samples")
    print(f"Samples per class (train): ~{len(train_labels) // N_CLASSES}")
    return train_labels, test_labels, train_class_inds, test_class_inds


def generate_ltp(train_labels, test_labels, train_class_inds, test_class_inds):
    """LTP mode: each client sees all 100 classes in a unique random order.
    10 tasks × 10 classes/task → CIFAR100_split_cn10_tn10_cet10_cs1_s2571.pkl
    """
    N_TASKS         = 10
    CLASSES_PER_TASK = 10
    save_path = os.path.join(SPLIT_DIR, "CIFAR100_split_cn10_tn10_cet10_cs1_s2571.pkl")

    train_inds, test_inds, client_y_list = [], [], []

    print(f"\nGenerating LTP split: {N_CLIENTS} clients × {N_TASKS} tasks × {CLASSES_PER_TASK} classes …")
    for client_id in range(N_CLIENTS):
        rng = np.random.RandomState(SEED + client_id * 137)
        all_classes = rng.permutation(N_CLASSES)

        client_train, client_test, client_labels = [], [], []
        for t in range(N_TASKS):
            task_cls = all_classes[t * CLASSES_PER_TASK:(t + 1) * CLASSES_PER_TASK].tolist()
            client_labels.append(task_cls)
            client_train.append([idx for c in task_cls for idx in train_class_inds[c]])
            client_test.append( [idx for c in task_cls for idx in test_class_inds[c]])

        train_inds.append(client_train)
        test_inds.append(client_test)
        client_y_list.append(client_labels)
        print(f"  Client {client_id}: task0 classes = {all_classes[:CLASSES_PER_TASK].tolist()}")

    # Verify
    print("\nVerifying LTP split …")
    for c_i in range(N_CLIENTS):
        seen = set()
        for t_i in range(N_TASKS):
            actual = set(train_labels[np.array(train_inds[c_i][t_i])].tolist())
            expected = set(client_y_list[c_i][t_i])
            assert actual == expected, f"Client {c_i}, Task {t_i}: mismatch"
            seen.update(expected)
        assert len(seen) == N_CLASSES, f"Client {c_i}: missing classes"
    print("  All checks passed ✓")

    return train_inds, test_inds, client_y_list, save_path, N_TASKS, CLASSES_PER_TASK


def generate_random(train_labels, test_labels, train_class_inds, test_class_inds):
    """Random-sampling mode: each client randomly picks 20 classes per task (may overlap).
    4 tasks × 20 classes/task → CIFAR100_split_cn10_tn4_cet20_s2571.pkl
    """
    N_TASKS          = 4
    CLASSES_PER_TASK = 20
    save_path = os.path.join(SPLIT_DIR, "CIFAR100_split_cn10_tn4_cet20_s2571.pkl")

    train_inds, test_inds, client_y_list = [], [], []

    print(f"\nGenerating random split: {N_CLIENTS} clients × {N_TASKS} tasks × {CLASSES_PER_TASK} classes/task …")
    print("(Each task randomly samples 20 classes from 100 — overlap allowed)")
    for client_id in range(N_CLIENTS):
        rng = np.random.RandomState(SEED + client_id * 137)

        client_train, client_test, client_labels = [], [], []
        client_all_classes = set()
        for t in range(N_TASKS):
            task_cls = sorted(rng.choice(N_CLASSES, size=CLASSES_PER_TASK, replace=False).tolist())
            client_labels.append(task_cls)
            client_all_classes.update(task_cls)
            client_train.append([idx for c in task_cls for idx in train_class_inds[c]])
            client_test.append( [idx for c in task_cls for idx in test_class_inds[c]])

        train_inds.append(client_train)
        test_inds.append(client_test)
        client_y_list.append(client_labels)
        print(f"  Client {client_id}: {len(client_all_classes)} unique classes, "
              f"task0 sample: {client_labels[0][:5]}…")

    return train_inds, test_inds, client_y_list, save_path, N_TASKS, CLASSES_PER_TASK


def main():
    parser = argparse.ArgumentParser(description="Generate CIFAR-100 FL split")
    parser.add_argument("--mode", choices=["ltp", "random"], default="ltp",
                        help="ltp: LTP 10×10 split (default); random: random-sample 4×20 split")
    args = parser.parse_args()

    os.makedirs(SPLIT_DIR, exist_ok=True)
    np.random.seed(SEED)

    train_labels, test_labels, train_class_inds, test_class_inds = load_cifar100()

    if args.mode == "ltp":
        result = generate_ltp(train_labels, test_labels, train_class_inds, test_class_inds)
    else:
        result = generate_random(train_labels, test_labels, train_class_inds, test_class_inds)

    train_inds, test_inds, client_y_list, save_path, n_tasks, cpt = result

    split_data = {
        "train_inds":    train_inds,
        "test_inds":     test_inds,
        "client_y_list": client_y_list,
    }
    with open(save_path, "wb") as f:
        pickle.dump(split_data, f)

    print(f"\nSplit file saved → {save_path}")
    print(f"  Clients: {N_CLIENTS}  |  Tasks: {n_tasks}  |  Classes/task: {cpt}")


if __name__ == "__main__":
    main()
