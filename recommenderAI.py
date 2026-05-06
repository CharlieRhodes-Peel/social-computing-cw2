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

N_EPOCHS    = 2       # Training passes over the full dataset.
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
            n += 1

    global_mean = total_rating / n
    print(f"  Users: {len(user_ids):,}  |  Items: {len(item_ids):,}  |  Ratings: {n:,}  |  μ={global_mean:.4f}")
    return user_ids, item_ids, global_mean


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


def _sgd_chunk(chunk, user_map, item_map, global_mean, P, Q, bu, bi, lr, total_sq_err, n_ratings):
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

def compute_mae(predictions, actuals):
    """
    Mean Absolute Error = (1/N) * Σ |r - r̂|

    MAE is preferred over RMSE for recommender evaluation because it treats
    all errors equally regardless of magnitude, matching how users experience
    inaccuracy (a 1-star error feels the same whether predicted 3 vs actual 4,
    or predicted 1 vs actual 2).
    """
    errors = [abs(p - a) for p, a in zip(predictions, actuals)]
    return sum(errors) / len(errors)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Matrix Factorisation Recommender — SGD/SVD")
    print("=" * 55)
    print(f"  Factors={N_FACTORS}, Epochs={N_EPOCHS}, lr={LEARNING_RATE}, reg={REGULARISATION}")
    print()

    # ── 1. First pass: discover dimensions and μ
    user_ids, item_ids, global_mean = first_pass(TRAIN_FILE)

    # ── 2. Build compact index mappings
    user_map, item_map = build_mappings(user_ids, item_ids)
    n_users = len(user_map)
    n_items = len(item_map)

    # ── 3. Initialise factor matrices and bias vectors
    print(f"Initialising model: {n_users:,} users × {n_items:,} items × {N_FACTORS} factors")
    P, Q, bu, bi = init_model(n_users, n_items, N_FACTORS)
    mem_mb = (P.nbytes + Q.nbytes + bu.nbytes + bi.nbytes) / 1024 / 1024
    print(f"  Parameter memory: {mem_mb:.1f} MB\n")

    # ── 4. Train
    print("Training...")
    train(TRAIN_FILE, user_map, item_map, global_mean, P, Q, bu, bi)
    print()

    # ── 5. Predict test set
    generate_predictions(TEST_FILE, OUTPUT_FILE, user_map, item_map, global_mean, P, Q, bu, bi)

    print("\nDone.")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    main()