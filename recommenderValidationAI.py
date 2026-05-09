"""
Matrix Factorisation Recommender System
========================================
Implements Simon Funk-style SVD with biases, trained via Stochastic Gradient Descent (SGD).

Theory
------
We model a predicted rating as:

    r̂(u, i) = μ + b_u + b_i + P[u] · Q[i]

Where:
    μ       = global mean rating (scalar)
    b_u     = user bias (how much user u rates above/below average)
    b_i     = item bias (how much item i is rated above/below average)
    P[u]    = latent factor vector for user u  (shape: n_factors,)
    Q[i]    = latent factor vector for item i  (shape: n_factors,)
    P[u]·Q[i] = dot product capturing user-item interaction

The error on a single known rating r(u,i) is:

    e(u,i) = r(u,i) - r̂(u,i)

We minimise the regularised squared error loss over all known ratings:

    L = Σ [ e(u,i)² + λ(b_u² + b_i² + ||P[u]||² + ||Q[i]||²) ]

Taking partial derivatives and applying gradient descent gives the SGD update rules:

    b_u  +=  η * (e - λ * b_u)
    b_i  +=  η * (e - λ * b_i)
    P[u] +=  η * (e * Q[i] - λ * P[u])
    Q[i] +=  η * (e * P[u] - λ * Q[i])        ← note: uses OLD P[u] before update

Where η = learning rate, λ = regularisation coefficient.

Note on update order: P[u] is updated before Q[i] in code, so we capture the
old P[u] value first to use in the Q[i] update. This avoids a subtle bias.
"""

import csv
import time
import random
import numpy as np


# ─────────────────────────────────────────────
# Hyperparameters — tune these for better MAE
# ─────────────────────────────────────────────
N_FACTORS   = 50      # Number of latent dimensions. More = expressive but slower + overfits.
                      # 50-200 is typical. Start at 50, increase if MAE is still high.

N_EPOCHS    = 20      # Training passes over the full dataset.
                      # Each extra epoch improves fit but risks overfitting.

LEARNING_RATE = 0.005 # Step size for gradient descent (η).
                      # Too high → diverges. Too low → slow convergence.

REGULARISATION = 0.02 # L2 penalty coefficient (λ).
                      # Prevents overfitting by penalising large factor values.
                      # 0.01–0.05 is a typical range.

LR_DECAY    = 0.96    # Multiply learning rate by this after each epoch.
                      # Allows large steps early, fine-tuning later.

RATING_CLIP = (0.5, 5.0)  # Valid rating range — clamp predictions to this.

TRAIN_FILE  = "train_20M_withratings.csv"
TEST_FILE   = "test_20M_withoutratings.csv"
OUTPUT_FILE = "predictions.csv"

CHUNK_SIZE  = 1_000_000   # Rows to read at once during training to manage RAM.

# ── Validation settings ──────────────────────────────────────────────────────
# When VALIDATION_MODE = True, the script splits the training file by users:
#   - 80% of users are used for training
#   - 20% of users are held out; their ratings are used to compute MAE
# This gives an honest estimate of how well the model generalises to users
# it has never seen (cold-start scenario).
#
# Splitting by user is intentionally stricter than splitting by random rows.
# A row-level split would let the same user appear in both train and val,
# giving the model pre-trained factor vectors for validation users — which
# would produce an unrealistically optimistic MAE.
#
# Set VALIDATION_MODE = False for your final submission run so all training
# data is used and predictions are written to OUTPUT_FILE as normal.
VALIDATION_MODE      = True
VALIDATION_SPLIT     = 0.2   # Fraction of users to hold out for validation


# ─────────────────────────────────────────────
# Step 1: First pass — collect unique IDs and global mean
# ─────────────────────────────────────────────

def first_pass(filepath):
    """
    Read through the training file once to:
      - Find all unique user IDs and item IDs
      - Compute the global mean rating μ
    We do this before allocating factor matrices so we know their sizes.
    Returns:
        user_ids (set), item_ids (set), global_mean (float)
    """
    print("First pass: collecting IDs and computing global mean...")
    user_ids = set()
    item_ids = set()
    row_count = set()
    total_rating = 0.0
    n = 0

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            user_ids.add(int(row[0]))
            item_ids.add(int(row[1]))
            total_rating += float(row[2])
            row_count.add(n)
            n += 1

    global_mean = total_rating / n
    print(f"  Users: {len(user_ids):,}  |  Items: {len(item_ids):,}  |  Ratings: {n:,}  |  μ={global_mean:.4f}")
    return user_ids, item_ids, global_mean, row_count


# ─────────────────────────────────────────────
# Step 2: Build compact ID mappings
# ─────────────────────────────────────────────

def build_mappings(user_ids, item_ids):
    """
    Raw IDs (e.g. user 138493) can't index directly into numpy arrays.
    We map each ID to a contiguous integer index [0, N).
    This keeps our factor matrices as small as possible.
    Returns:
        user_map  {raw_id: index}
        item_map  {raw_id: index}
    """
    user_map = {uid: idx for idx, uid in enumerate(sorted(user_ids))}
    item_map = {iid: idx for idx, iid in enumerate(sorted(item_ids))}
    return user_map, item_map


# ─────────────────────────────────────────────
# Step 2b: Split users into train / validation sets
# ─────────────────────────────────────────────

def split_rows(row_count, val_fraction=0.2, seed=42):
    """
    Randomly assign rows to train or validation groups.

    Returns:
        train_rows (set), val_rows (set)
    """
    rng = random.Random(seed)
    row_list = sorted(row_count)
    rng.shuffle(row_list)

    split_idx   = int(len(row_list) * (1 - val_fraction))
    train_rows = set(row_list[:split_idx])
    val_rows   = set(row_list[split_idx:])

    print(f"  Train rows: {len(train_rows):,}  |  Val rows: {len(val_rows):,}  ({val_fraction*100:.0f}% held out)")
    return train_rows, val_rows


def partition_ratings(filepath, train_rows, val_rows):
    """
    Read the training file once and split rows into:
      - train_ratings: list of (user_raw, item_raw, rating) for train users
      - val_ratings:   list of (user_raw, item_raw, rating) for val users

    Note: This loads both sets into RAM. At 20M rows this is approximately
    2–3 GB. If RAM is a concern, reduce CHUNK_SIZE or write val_ratings to
    a temp file instead of holding them in memory.

    An alternative memory-efficient approach would be to read the file twice
    (once for train, once for val), but two passes is slower on spinning disk.
    """
    print("Partitioning ratings into train / val sets...")
    train_ratings = []
    val_ratings   = []

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            if not row:
                continue

            u = int(row[0])
            i = int(row[1])
            r = float(row[2])
            if count in train_rows:
                train_ratings.append((u, i, r))
            elif count in val_rows:
                val_ratings.append((u, i, r))

            count = count + 1

    print(f"  Train ratings: {len(train_ratings):,}  |  Val ratings: {len(val_ratings):,}")
    return train_ratings, val_ratings


# ─────────────────────────────────────────────
# Step 3: Initialise model parameters
# ─────────────────────────────────────────────

def init_model(n_users, n_items, n_factors):
    """
    Initialise all learnable parameters.

    Factor matrices P and Q are initialised with small random values drawn
    from a normal distribution with std=0.1. This breaks symmetry — if all
    factors started at zero, all gradients would be equal and P≡Q would stay
    zero forever (the 'cold start symmetry' problem in matrix factorisation).

    Biases start at zero: they have no symmetry problem because they are
    scalar per-entity terms, not vectors.

    Returns:
        P  (n_users × n_factors)  user latent factors
        Q  (n_items × n_factors)  item latent factors
        bu (n_users,)             user biases
        bi (n_items,)             item biases
    """
    P  = np.random.normal(0.0, 0.1, (n_users, n_factors)).astype(np.float32)
    Q  = np.random.normal(0.0, 0.1, (n_items, n_factors)).astype(np.float32)
    bu = np.zeros(n_users, dtype=np.float32)
    bi = np.zeros(n_items, dtype=np.float32)
    return P, Q, bu, bi


# ─────────────────────────────────────────────
# Step 4: Train — SGD over all ratings
# ─────────────────────────────────────────────

def train(filepath, user_map, item_map, global_mean, P, Q, bu, bi):
    """
    Train the model using Stochastic Gradient Descent (SGD).

    We process the training file in chunks of CHUNK_SIZE rows to avoid loading
    20M rows into RAM simultaneously. Within each chunk we shuffle the order
    before processing — SGD converges better when updates aren't correlated by
    user (which they would be if we processed user 1's ratings all together,
    then user 2's, etc.).

    Each rating produces one set of gradient updates. After all epochs, P, Q,
    bu, bi are mutated in-place and reflect the trained model.
    """
    lr = LEARNING_RATE

    for epoch in range(N_EPOCHS):
        t0 = time.time()
        total_sq_err = 0.0
        n_ratings = 0

        with open(filepath, "r") as f:
            reader = csv.reader(f)
            chunk = []

            for row in reader:
                if not row:
                    continue
                chunk.append(row)

                if len(chunk) >= CHUNK_SIZE:
                    total_sq_err, n_ratings = _sgd_chunk(
                        chunk, user_map, item_map, global_mean,
                        P, Q, bu, bi, lr, total_sq_err, n_ratings
                    )
                    chunk = []

            # Process any remaining rows in the last partial chunk
            if chunk:
                total_sq_err, n_ratings = _sgd_chunk(
                    chunk, user_map, item_map, global_mean,
                    P, Q, bu, bi, lr, total_sq_err, n_ratings
                )

        rmse = (total_sq_err / n_ratings) ** 0.5
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:2d}/{N_EPOCHS}  |  RMSE={rmse:.4f}  |  lr={lr:.5f}  |  {elapsed:.1f}s")

        # Decay learning rate each epoch — take smaller steps as we approach convergence
        lr *= LR_DECAY


def train_from_list(ratings_list, user_map, item_map, global_mean, P, Q, bu, bi):
    """
    Training loop used in VALIDATION_MODE. Accepts a pre-loaded list of
    (user_raw, item_raw, rating) tuples rather than reading from file.
    The list has already been partitioned to exclude val users.
    """
    lr = LEARNING_RATE

    for epoch in range(N_EPOCHS):
        t0 = time.time()
        total_sq_err = 0.0
        n_ratings = 0

        # Shuffle the full list each epoch so SGD updates aren't user-ordered
        random.shuffle(ratings_list)

        # Process in chunks to keep the loop structure consistent
        for start in range(0, len(ratings_list), CHUNK_SIZE):
            chunk = ratings_list[start : start + CHUNK_SIZE]
            total_sq_err, n_ratings = _sgd_chunk_from_tuples(
                chunk, user_map, item_map, global_mean,
                P, Q, bu, bi, lr, total_sq_err, n_ratings
            )

        rmse = (total_sq_err / n_ratings) ** 0.5
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:2d}/{N_EPOCHS}  |  RMSE={rmse:.4f}  |  lr={lr:.5f}  |  {elapsed:.1f}s")
        lr *= LR_DECAY


def _sgd_chunk_from_tuples(chunk, user_map, item_map, global_mean, P, Q, bu, bi, lr, total_sq_err, n_ratings):
    """
    SGD update loop operating on pre-parsed (user_raw, item_raw, rating) tuples.
    Identical maths to _sgd_chunk — separated only to avoid re-parsing CSV strings.
    See _sgd_chunk for full derivation commentary.
    """
    reg = REGULARISATION
    for (u_raw, i_raw, r) in chunk:
        if u_raw not in user_map or i_raw not in item_map:
            continue
        u = user_map[u_raw]
        i = item_map[i_raw]

        pred = global_mean + bu[u] + bi[i] + np.dot(P[u], Q[i])
        e    = r - pred

        bu[u] += lr * (e - reg * bu[u])
        bi[i] += lr * (e - reg * bi[i])

        p_u_old  = P[u].copy()
        P[u]    += lr * (e * Q[i]    - reg * P[u])
        Q[i]    += lr * (e * p_u_old - reg * Q[i])

        total_sq_err += e * e
        n_ratings    += 1

    return total_sq_err, n_ratings



    """
    Apply SGD updates for one chunk of ratings (shuffled).

    The update equations (derived from ∂L/∂parameter = 0 via chain rule):

        e    = r - (μ + b_u + b_i + P[u]·Q[i])

        b_u += η * (e - λ * b_u)      ← gradient of L w.r.t. b_u
        b_i += η * (e - λ * b_i)      ← gradient of L w.r.t. b_i

        # Save P[u] BEFORE updating it, for use in Q[i] update
        p_u_old = P[u].copy()
        P[u] += η * (e * Q[i] - λ * P[u])    ← gradient of L w.r.t. P[u]
        Q[i] += η * (e * p_u_old - λ * Q[i]) ← gradient of L w.r.t. Q[i]

    Using p_u_old ensures both P and Q are updated with respect to the
    SAME error e, rather than e recalculated after P[u] changed.
    """
    random.shuffle(chunk)
    reg = REGULARISATION

    for row in chunk:
        u_raw = int(row[0])
        i_raw = int(row[1])
        r     = float(row[2])

        # Skip any user/item not seen during first pass (shouldn't happen in training set)
        if u_raw not in user_map or i_raw not in item_map:
            continue

        u = user_map[u_raw]
        i = item_map[i_raw]

        # Prediction and error
        pred = global_mean + bu[u] + bi[i] + np.dot(P[u], Q[i])
        e = r - pred

        # Bias updates (simple scalar gradient step)
        bu[u] += lr * (e - reg * bu[u])
        bi[i] += lr * (e - reg * bi[i])

        # Factor updates — capture old P[u] first!
        p_u_old = P[u].copy()
        P[u] += lr * (e * Q[i]     - reg * P[u])
        Q[i] += lr * (e * p_u_old  - reg * Q[i])

        total_sq_err += e * e
        n_ratings    += 1

    return total_sq_err, n_ratings


# ─────────────────────────────────────────────
# Step 5: Predict
# ─────────────────────────────────────────────

def predict_rating(u_idx, i_idx, global_mean, P, Q, bu, bi):
    """
    Compute r̂(u, i) = μ + b_u + b_i + P[u]·Q[i]

    For users/items not seen during training (cold-start), we fall back to
    simpler estimates:
      - Unknown user AND item → global mean μ
      - Unknown user, known item → μ + b_i
      - Known user, unknown item → μ + b_u

    Predictions are clamped to [0.5, 5.0] since ratings outside this range
    are not meaningful and would inflate MAE.
    """
    if u_idx is None and i_idx is None:
        pred = global_mean
    elif u_idx is None:
        pred = global_mean + bi[i_idx]
    elif i_idx is None:
        pred = global_mean + bu[u_idx]
    else:
        pred = global_mean + bu[u_idx] + bi[i_idx] + np.dot(P[u_idx], Q[i_idx])

    return float(np.clip(pred, RATING_CLIP[0], RATING_CLIP[1]))


# ─────────────────────────────────────────────
# Step 6: Generate predictions for test set
# ─────────────────────────────────────────────

def generate_predictions(test_filepath, output_filepath, user_map, item_map, global_mean, P, Q, bu, bi):
    """
    Read the test file and write a prediction for every (user, item) pair.
    Test file format: user_id, item_id, timestamp (no rating column).
    Output format: user_id, item_id, predicted_rating
    """
    print(f"Generating predictions → {output_filepath}")
    n = 0

    with open(test_filepath, "r") as fin, open(output_filepath, "w", newline="") as fout:
        reader  = csv.reader(fin)
        writer  = csv.writer(fout)

        for row in reader:
            if not row:
                continue
            u_raw = int(row[0])
            i_raw = int(row[1])

            u_idx = user_map.get(u_raw)   # None if unseen user
            i_idx = item_map.get(i_raw)   # None if unseen item

            pred = predict_rating(u_idx, i_idx, global_mean, P, Q, bu, bi)
            writer.writerow([u_raw, i_raw, f"{pred:.4f}"])
            n += 1

    print(f"  Wrote {n:,} predictions.")


# ─────────────────────────────────────────────
# Step 7 (optional): Evaluate on a held-out split
# ─────────────────────────────────────────────

def evaluate_validation(val_ratings, user_map, item_map, global_mean, P, Q, bu, bi):
    """
    Compute MAE on the held-out validation users.

    MAE = (1/N) * Σ |r(u,i) - r̂(u,i)|

    We also report:
      - RMSE   = sqrt( (1/N) * Σ e² )   — penalises large errors more heavily
      - Coverage — what % of val pairs we could predict (vs cold-start fallback)
      - Per-bucket MAE — breaks down error by true rating value so you can see
        where the model struggles (typically the extremes: 0.5 and 5.0)

    All val users are cold-start (unseen during training) so their factor
    vectors P[u] are uninitialised. We fall back to μ + b_i for them.
    Seen items still contribute their item bias b_i, which helps considerably.
    """
    print("\nEvaluating on validation set...")
    abs_errors  = []
    sq_errors   = []
    cold_count  = 0        # predictions that fell back to μ or μ+b_i
    bucket_errors = {}     # {true_rating: [abs_errors]}

    for (u_raw, i_raw, r_true) in val_ratings:
        u_idx = user_map.get(u_raw)   # Will be None — val users unseen in training
        i_idx = item_map.get(i_raw)

        pred = predict_rating(u_idx, i_idx, global_mean, P, Q, bu, bi)

        if u_idx is None:
            cold_count += 1

        ae = abs(r_true - pred)
        abs_errors.append(ae)
        sq_errors.append(ae * ae)

        # Bucket by true rating (rounded to nearest 0.5)
        bucket = round(r_true * 2) / 2
        if bucket not in bucket_errors:
            bucket_errors[bucket] = []
        bucket_errors[bucket].append(ae)

    n         = len(abs_errors)
    mae       = sum(abs_errors) / n
    rmse      = (sum(sq_errors) / n) ** 0.5
    cold_pct  = 100.0 * cold_count / n

    print(f"\n{'─'*40}")
    print(f"  Validation results ({n:,} ratings)")
    print(f"{'─'*40}")
    print(f"  MAE   : {mae:.4f}   ← primary metric")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  Cold-start predictions: {cold_pct:.1f}%  (val users have no learned factors)")
    print(f"\n  MAE by true rating bucket:")
    for bucket in sorted(bucket_errors.keys()):
        errs    = bucket_errors[bucket]
        b_mae   = sum(errs) / len(errs)
        bar     = "█" * int(b_mae * 30)
        print(f"    {bucket:.1f}  {bar:<30}  {b_mae:.4f}  (n={len(errs):,})")
    print(f"{'─'*40}\n")

    return mae


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Matrix Factorisation Recommender — SGD/SVD")
    print("=" * 55)
    print(f"  Factors={N_FACTORS}, Epochs={N_EPOCHS}, lr={LEARNING_RATE}, reg={REGULARISATION}")
    print(f"  Mode: {'VALIDATION (80/20 user split)' if VALIDATION_MODE else 'FULL TRAINING'}")
    print()

    # ── 1. First pass: discover all IDs and global mean
    user_ids, item_ids, global_mean, row_count = first_pass(TRAIN_FILE)

    # ── 2. Build compact index mappings
    user_map, item_map = build_mappings(user_ids, item_ids)
    n_users = len(user_map)
    n_items = len(item_map)

    # ── 3. Initialise factor matrices and bias vectors
    print(f"Initialising model: {n_users:,} users × {n_items:,} items × {N_FACTORS} factors")
    P, Q, bu, bi = init_model(n_users, n_items, N_FACTORS)
    mem_mb = (P.nbytes + Q.nbytes + bu.nbytes + bi.nbytes) / 1024 / 1024
    print(f"  Parameter memory: {mem_mb:.1f} MB\n")

    if VALIDATION_MODE:
        # ── VALIDATION PATH ──────────────────────────────────────────────────
        # Split users 80/20 and partition all ratings accordingly
        train_rows, val_rows = split_rows(row_count, val_fraction=VALIDATION_SPLIT)
        train_ratings, val_ratings = partition_ratings(TRAIN_FILE, train_rows, val_rows)

        # Train only on the 80% train rows
        print("\nTraining on 80% of rows...")
        train_from_list(train_ratings, user_map, item_map, global_mean, P, Q, bu, bi)

        # Evaluate MAE on the held-out 20% val users
        evaluate_validation(val_ratings, user_map, item_map, global_mean, P, Q, bu, bi)

        print("Validation run complete. Set VALIDATION_MODE = False for final submission.")

    else:
        # ── FULL TRAINING PATH ───────────────────────────────────────────────
        # Train on all data, then write predictions for the real test set
        print("Training on full dataset...")
        train(TRAIN_FILE, user_map, item_map, global_mean, P, Q, bu, bi)
        print()
        generate_predictions(TEST_FILE, OUTPUT_FILE, user_map, item_map, global_mean, P, Q, bu, bi)
        print("\nDone. Submit:", OUTPUT_FILE)



if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    main()