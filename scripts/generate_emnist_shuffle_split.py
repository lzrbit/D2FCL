"""
Generator for EMNIST-Letters federated continual learning splits.

Produces TWO split files from the same class assignments:

  1. EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl          (LTP / non-shuffle)
     → Classes assigned to tasks in SORTED order (same global ordering for all clients)

  2. EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl  (Shuffle)
     → Classes assigned to tasks in independently RANDOMIZED order per client

Both files share the same per-client class subsets and sample indices,
so the ONLY difference is the temporal class arrival order.
"""

import os
import pickle
import numpy as np
from torchvision import datasets, transforms

# ── Configuration ────────────────────────────────────────────────────────────
N_CLIENTS         = 8
N_TASKS           = 6
CLASSES_PER_TASK  = 2      # each client sees 2 new classes per task
N_EMNIST_CLASSES  = 26     # EMNIST-Letters: a–z (labels 0–25 after transform)
CLASSES_PER_CLIENT = N_TASKS * CLASSES_PER_TASK   # = 12
SAMPLES_PER_CLASS  = 500   # training samples per class per client
SEED               = 2571

DATA_DIR   = "datasets"
SPLIT_DIR  = os.path.join(DATA_DIR, "split_files")
LTP_PATH     = os.path.join(SPLIT_DIR, "EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl")
SHUFFLE_PATH = os.path.join(SPLIT_DIR, "EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl")

os.makedirs(SPLIT_DIR, exist_ok=True)

# ── Load EMNIST Letters ───────────────────────────────────────────────────────
print("Loading EMNIST Letters …")
emnist_train = datasets.EMNIST(DATA_DIR, split='letters', train=True,
                                download=False, transform=transforms.ToTensor(),
                                target_transform=lambda x: x - 1)   # labels → 0-25
emnist_test  = datasets.EMNIST(DATA_DIR, split='letters', train=False,
                                download=False, transform=transforms.ToTensor(),
                                target_transform=lambda x: x - 1)

train_labels = np.array(emnist_train.targets) - 1
test_labels  = np.array(emnist_test.targets)  - 1

# Collect indices per class
train_class_inds = {c: np.where(train_labels == c)[0].tolist() for c in range(N_EMNIST_CLASSES)}
test_class_inds  = {c: np.where(test_labels  == c)[0].tolist() for c in range(N_EMNIST_CLASSES)}

# ── Step 1: Determine class subsets and sample indices per client ─────────────
# (shared between LTP and Shuffle)
client_class_sets = []      # list of 12 classes per client (unsorted)
client_sample_inds = []     # dict: class -> (train_inds, test_inds) per client

print(f"Generating splits for {N_CLIENTS} clients × {N_TASKS} tasks × {CLASSES_PER_TASK} classes …")

for client_id in range(N_CLIENTS):
    rng = np.random.RandomState(SEED + client_id * 137)
    client_classes = rng.choice(N_EMNIST_CLASSES, size=CLASSES_PER_CLIENT, replace=False)
    client_class_sets.append(client_classes)

    samples = {}
    for c in client_classes:
        avail = train_class_inds[c]
        n_sample = min(SAMPLES_PER_CLASS, len(avail))
        chosen = rng.choice(avail, n_sample, replace=False).tolist()
        samples[c] = (chosen, test_class_inds[c])
    client_sample_inds.append(samples)

    print(f"  Client {client_id}: classes = {sorted(client_classes.tolist())}")


def build_split(client_class_sets, client_sample_inds, mode='ltp'):
    """Build split data with either 'ltp' (sorted) or 'shuffle' ordering."""
    train_inds    = []
    test_inds     = []
    client_y_list = []

    for client_id in range(N_CLIENTS):
        classes = client_class_sets[client_id].copy()

        if mode == 'ltp':
            # LTP: sort classes → deterministic global ordering for all clients
            classes = np.sort(classes)
        else:
            # Shuffle: independently random ordering per client
            # Use a separate RNG to avoid correlation with sample selection
            shuffle_rng = np.random.RandomState(SEED * 3 + client_id * 251)
            shuffle_rng.shuffle(classes)

        samples = client_sample_inds[client_id]
        client_train = []
        client_test  = []
        client_tasks = []

        for t in range(N_TASKS):
            task_classes = classes[t * CLASSES_PER_TASK: (t + 1) * CLASSES_PER_TASK].tolist()
            client_tasks.append(task_classes)

            task_train = []
            task_test  = []
            for c in task_classes:
                tr, te = samples[c]
                task_train.extend(tr)
                task_test.extend(te)

            client_train.append(task_train)
            client_test.append(task_test)

        train_inds.append(client_train)
        test_inds.append(client_test)
        client_y_list.append(client_tasks)

    return {"train_inds": train_inds, "test_inds": test_inds, "client_y_list": client_y_list}


# ── Step 2: Generate LTP split ───────────────────────────────────────────────
print("\n── LTP (non-shuffle) split ──")
ltp_data = build_split(client_class_sets, client_sample_inds, mode='ltp')
for i, cy in enumerate(ltp_data['client_y_list']):
    flat = [c for pair in cy for c in pair]
    print(f"  Client {i}: task order = {flat}")

# ── Step 3: Generate Shuffle split ───────────────────────────────────────────
print("\n── Shuffle split ──")
shuffle_data = build_split(client_class_sets, client_sample_inds, mode='shuffle')
for i, cy in enumerate(shuffle_data['client_y_list']):
    flat = [c for pair in cy for c in pair]
    print(f"  Client {i}: task order = {flat}")

# ── Verify both splits ──────────────────────────────────────────────────────
print("\nVerifying …")
for name, split_data in [("LTP", ltp_data), ("Shuffle", shuffle_data)]:
    for c_i in range(N_CLIENTS):
        for t_i in range(N_TASKS):
            actual = set(train_labels[np.array(split_data['train_inds'][c_i][t_i])].tolist())
            expected = set(split_data['client_y_list'][c_i][t_i])
            assert actual == expected, \
                f"{name} Client {c_i}, Task {t_i}: label mismatch {actual} != {expected}"
    # Verify same class sets (only ordering differs)
    for c_i in range(N_CLIENTS):
        ltp_flat = sorted(c for pair in ltp_data['client_y_list'][c_i] for c in pair)
        shf_flat = sorted(c for pair in shuffle_data['client_y_list'][c_i] for c in pair)
        assert ltp_flat == shf_flat, f"Client {c_i}: class sets differ!"
print("  All checks passed ✓")

# Verify orderings are actually different
n_diff = sum(
    1 for i in range(N_CLIENTS)
    if ltp_data['client_y_list'][i] != shuffle_data['client_y_list'][i]
)
print(f"  Clients with different task ordering: {n_diff}/{N_CLIENTS}")
assert n_diff > 0, "LTP and Shuffle orderings are identical — bug!"

# ── Save ─────────────────────────────────────────────────────────────────────
with open(LTP_PATH, "wb") as f:
    pickle.dump(ltp_data, f)
print(f"\nLTP split saved     → {LTP_PATH}")

with open(SHUFFLE_PATH, "wb") as f:
    pickle.dump(shuffle_data, f)
print(f"Shuffle split saved → {SHUFFLE_PATH}")

print(f"\n  Clients: {N_CLIENTS}  |  Tasks: {N_TASKS}  |  Classes/task: {CLASSES_PER_TASK}")
print(f"  Train samples/class: {SAMPLES_PER_CLASS}  |  Seed: {SEED}")