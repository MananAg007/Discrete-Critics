# Binary Success Critic (BSC) for FQL — Design Document

## 1. Motivation

FQL's critic is trained with MSE regression to the Bellman target:

```
target_q = r + γ * mask * Q_target(s', a')
```

In the AntMaze OGBench environment, rewards are sparse binary (1 at goal, 0 otherwise). After D4RL processing (`d4rl_utils.py`), rewards are shifted:

```
reward = original_reward - 1.0
# success step:   reward = 0,  mask = 0
# all other steps: reward = -1, mask = 1
```

The MSE critic learns `Q(s,a) ≈ -1 * (1 + γ + ... + γ^(t-1)) + 0` for non-success trajectories and `Q(s,a) ≈ 0` at the success terminal step. For states along a successful trajectory of length t:

```
Q(s_0, a_0) ≈ −(1 + γ + ... + γ^(t-1)) * (-1) = (1 - γ^t) / (1 - γ)
```

This implicitly encodes **time-to-success** — shorter paths receive higher Q-values — even though the actual evaluation metric is purely binary success rate. The optimizer is therefore spending capacity modeling "how fast" rather than "whether".

The Binary Success Critic (BSC) replaces this with a classifier: **`Q(s,a) ∈ [0,1]` estimates the probability that taking action `a` at state `s` leads to task success within the episode**, trained with binary cross-entropy loss. The TD targets encode success probability directly.

---

## 2. Dataset Structure Analysis

### D4RL AntMaze (d4rl_utils.py)

The `get_dataset` function in `envs/d4rl_utils.py` produces the following buffer fields:

```python
terminals[i] = float(
    np.linalg.norm(obs[i+1] - next_obs[i]) > 1e-6
)  # 1 at ANY trajectory boundary (success or timeout)

masks[i] = 1 - dataset['terminals'][i]
# dataset['terminals'] from D4RL AntMaze = 1 only on goal-reaching success
# So: mask = 0 ONLY at success, mask = 1 at timeout and non-terminal
```

These three buffer fields (already in the dataset) uniquely identify all transition types:

| Case | `masks` | `terminals` | `terminals * masks` | Reward | BSC target |
|------|---------|-------------|---------------------|--------|------------|
| Success (`terminated=True`) | 0 | 1 | 0 | 0 | **1.0** |
| Timeout (`truncated=True`) | 1 | 1 | **1** | -1 | **0.0** |
| Non-terminal | 1 | 0 | 0 | -1 | **Q_target(s', a')** |

`truncated_flag = terminals * masks` uniquely identifies timeout steps.

**Critical note**: The `terminals` field in this dataset is **NOT the same as `masks`**:
- `masks = 1 - dataset['terminals']`: D4RL success flag (goal-reaching only)
- `terminals` (FQL's): observation-jump detector; 1 for ALL trajectory boundaries

**No new buffer fields are needed.** The BSC target is fully derivable from existing `masks` and `terminals`.

### OGBench (singletask environments)

OGBench provides properly formatted datasets from `ogbench.make_env_and_datasets`. The expected structure is analogous:
- `masks = 0` for success (terminated), `masks = 1` otherwise
- `terminals = 1` for trajectory boundaries (both success and timeout)

The same BSC formula applies. **Verify that OGBench preserves the success vs. timeout distinction in `terminals`** before enabling BSC on OGBench environments.

---

## 3. BSC Target Formula

With `γ = 1.0` (see §7), the BSC Bellman target is:

```
truncated = terminals * masks   # 1 only at timeout
target_bsc = (1 - masks) * 1.0  +  masks * (1 - truncated) * Q_target(s', a')
```

Expanding per case:
- `masks = 0` (success): `target = 1.0`
- `masks = 1, terminals = 1` (timeout): `target = 0.0`
- `masks = 1, terminals = 0` (non-terminal): `target = Q_target(s', a') ∈ [0, 1]`

The reward signal is **entirely unused** in the BSC target. This eliminates:
- The D4RL reward offset of -1 (which was only a convention choice anyway)
- Any offline/online reward mismatch when transitioning to the online phase
- Sensitivity to reward scale in the actor Q-loss (bounded Q ∈ [0,1])

---

## 4. Changes to the Critic Network

### 4.1 Output Head

The `Value` class in `utils/networks.py` currently produces an unbounded scalar Q-value via a linear output layer. For BSC, a sigmoid activation is added:

**Currently:**
```python
# Value.__call__: output is unbounded scalar
q = nn.Dense(1)(x)  # shape: (num_ensembles, B, 1)
return q.squeeze(-1)  # (num_ensembles, B)
```

**BSC change:**
```python
# Produce logit for BCE loss, then apply sigmoid
q_logit = nn.Dense(1)(x).squeeze(-1)   # (num_ensembles, B) — unbounded
q_prob  = jax.nn.sigmoid(q_logit)      # (num_ensembles, B) — ∈ [0, 1]
if return_logits:
    return q_prob, q_logit
return q_prob
```

The `return_logits=False` default means all existing call sites (action selection, Q logging, actor gradient) receive probabilities unchanged. Only the critic loss path passes `return_logits=True`.

**Why logits for the loss?** `optax.sigmoid_binary_cross_entropy(logit, label)` computes `log(σ(logit))` in a numerically stable way, avoiding catastrophic cancellation when `logit` is large in magnitude.

### 4.2 Target Critic

The target critic is also a `Value` module (soft-copy of the online critic). When BSC is enabled, it also produces probabilities via sigmoid. No separate treatment needed — the target critic's Q-output goes directly into `target_bsc` as the bootstrap value, already in [0,1].

---

## 5. Changes to the Critic Loss

### Current FQL critic loss (fql.py):

```python
def critic_loss(self, batch, grad_params, rng):
    next_actions = self.sample_actions(batch['next_observations'], seed=...)
    next_qs = self.network.select('target_critic')(batch['next_observations'], next_actions)
    next_q = next_qs.min(axis=0) if q_agg == 'min' else next_qs.mean(axis=0)

    target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q  # MSE target

    q = self.network.select('critic')(batch['observations'], batch['actions'], params=grad_params)
    critic_loss = jnp.square(q - target_q)  # MSE loss
    return critic_loss.mean(), {'critic_loss': ..., 'q': q.mean()}
```

### BSC critic loss:

```python
def critic_loss(self, batch, grad_params, rng):
    next_actions = self.sample_actions(batch['next_observations'], seed=...)
    next_qs = self.network.select('target_critic')(batch['next_observations'], next_actions)
    next_q = next_qs.min(axis=0) if q_agg == 'min' else next_qs.mean(axis=0)  # ∈ [0, 1]

    # BSC target: success=1, timeout=0, non-terminal=bootstrap
    truncated = batch['terminals'] * batch['masks']  # 1 only at timeout
    target_q = (1 - batch['masks']) * 1.0 + batch['masks'] * (1 - truncated) * next_q  # ∈ [0, 1]

    # Get logits for numerically stable BCE
    q_prob, q_logit = self.network.select('critic')(
        batch['observations'], batch['actions'],
        params=grad_params, return_logits=True
    )  # q_logit: (num_ensembles, B)

    # Binary cross-entropy (broadcast target across ensembles)
    critic_loss = optax.sigmoid_binary_cross_entropy(q_logit, target_q[None])
    return critic_loss.mean(), {
        'critic_loss': critic_loss.mean(),
        'q': q_prob.mean(),
        'target_q_mean': target_q.mean(),
        'target_q_success_frac': (1 - batch['masks']).mean(),
    }
```

---

## 6. Changes to the Actor Loss

The actor Q-loss component in `fql.py` is:

```python
qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
q = jnp.mean(qs, axis=0)   # (B,)
q_loss = -q.mean()
if self.config['normalize_q_loss']:
    lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
    q_loss = lam * q_loss

actor_loss = bc_flow_loss.mean() + alpha * distill_loss + q_loss
```

With BSC, `q ∈ [0, 1]` and the actor's total loss becomes:

```
actor_loss = bc_flow_loss + alpha * distill_loss - q.mean()
```

where `-q.mean() ∈ [-1, 0]`. The scale of `q_loss` is bounded.

### Key consideration: `normalize_q_loss`

Currently `normalize_q_loss=False` is the default for FQL. If enabled with BSC:
```python
lam = 1 / |q|.mean()
```
With BSC, `q` is a probability. Early in training on AntMaze (low success probability), `q.mean()` can be very small (e.g., 0.01), making `lam = 100` and exploding the Q-loss gradient. **Recommendation: keep `normalize_q_loss=False` for BSC**, or replace the normalization with a fixed scale.

### `alpha` coefficient re-tuning

The current `alpha=10` (BC distillation weight) was tuned against an unbounded Q-loss. With BSC, the Q-loss is bounded to `[-1, 0]`, so the BC and distillation losses may dominate. Consider **sweeping alpha** in the range `[0.5, 5.0]` rather than `[5, 100]`.

---

## 7. Discount Factor: γ = 1.0

For BSC to have the exact semantics of **P(eventual success | s, a)**:
- γ = 1.0: Every state on a successful trajectory converges to Q = 1.0
- γ < 1.0: Earlier states on a successful trajectory get lower Q (Q ≈ γ^t), reintroducing time-to-success bias

**Set `discount = 1.0` when using BSC.**

With γ=1.0, the timeout case (target=0.0, hard label, no bootstrap) is automatically handled correctly — timeout is treated as an absorbing failure state with value 0.

---

## 8. What Does NOT Change

- **Actor BC flow loss**: `bc_flow_loss = (pred_vel - vel)^2` — unchanged
- **Actor distillation loss**: `distill_loss = (actor_1step - bc_flow)^2` — unchanged
- **Replay buffer schema**: No new fields needed; uses existing `terminals` and `masks`
- **Target network soft update**: `tau=0.005` — unchanged
- **Ensemble size** (`value_num_ensembles=2`): Q-aggregation (min/mean) works identically on probabilities
- **Evaluation**: Already measures success rate, not discounted return — perfectly aligned
- **OGBench dataset creation**: OGBench provides pre-formatted datasets; no additional processing needed

---

## 9. Benefits

### 9.1 Alignment with Evaluation Metric
The evaluation metric for AntMaze is **success rate** — a binary signal. BSC trains Q directly as P(success), eliminating the mismatch between the implicit optimization objective (discounted path-length-penalized return) and the explicit evaluation metric.

### 9.2 Bounded Q-Values → Stable Training
MSE with unbounded Q-values can suffer from overestimation cascades: Q targets that are too large get regressed to, bootstrapping creates larger targets, etc. With sigmoid output, Q ∈ [0, 1] by construction. Overestimation is physically bounded by 1.0, preventing the typical offline RL Q-value explosion.

### 9.3 Eliminates Reward Engineering
The BSC target depends only on the binary `masks` (success/not) and `terminals` (boundary/not), ignoring rewards entirely. The reward offset of -1 in AntMaze preprocessing becomes irrelevant. This is especially valuable for:
- Offline-to-online transfer: no reward scale mismatch between phases
- Multi-task settings: no per-task reward normalization needed

### 9.4 Principled Handling of Timeouts
Standard FQL uses `mask=0` only for success, but bootstraps through timeout transitions (mask=1, terminal=1). This is strictly incorrect: the bootstrapped value `Q_target(next_obs)` at a timeout boundary corresponds to the NEXT episode's initial state, not the true continuation. BSC explicitly assigns `target=0.0` to timeout transitions using `truncated = terminals * masks`, eliminating this bootstrap error.

### 9.5 Better Offline Dataset Utilization
In AntMaze, the offline dataset contains many failed trajectories (never reach the goal). These contribute:
- In MSE training: negative Q-values propagating backward, weakly informative
- In BSC training: explicit `target=0.0` at trajectory boundaries, providing **direct gradient signal** that states on failed paths have zero success probability

---

## 10. Downsides and Risks

### 10.1 Vanishing Gradients in Sigmoid
When the critic is very confident (probabilities near 0 or 1), the sigmoid gradient `σ(z)(1-σ(z))` approaches 0. This can slow learning when:
- Transitions near the goal have near-certain success: Q ≈ 1, gradients → 0
- Transitions deep in failed trajectories have near-zero success: Q ≈ 0, gradients → 0

Mitigation: label smoothing on BSC targets (e.g., replace `1.0 → 0.95`, `0.0 → 0.05`) to keep some gradient flowing at the extremes.

### 10.2 Loss of Implicit Path-Length Signal
With MSE + γ=0.99, the Q-function implicitly rewards shorter paths. In sparse-reward navigation (AntMaze), this acts as a form of dense shaping: states closer to the goal in time get higher Q. BSC with γ=1.0 removes this signal. In theory this is desirable (alignment), but in practice this implicit shaping may have been helping with credit assignment and exploration during offline learning.

**Potential mitigation**: Experiment with γ=0.99 vs γ=1.0 for BSC. With γ<1.0, BSC values lose exact probability interpretation but retain the path-length signal. The BCE loss remains valid as long as targets stay in [0,1], which γ<1.0 with clipping achieves.

### 10.3 Slow Convergence on Long Horizons with γ=1.0
With γ=1.0, Q(s₀) for the initial state of a long trajectory equals Q(s_T) (the terminal state). The value doesn't decay with distance, so the critic must propagate success signals across many steps without discounting as a contraction. This can slow convergence compared to γ<1.0.

### 10.4 Actor Q-Loss Scale and Alpha Re-tuning
The BC distillation weight `alpha=10` was tuned for unbounded Q-values. With BSC q ∈ [0,1], the q_loss term is scaled down by roughly `1/(original Q scale)`. The `alpha` hyperparameter needs to be re-tuned. Expect optimal alpha to be lower than 10.

### 10.5 Ensemble Aggregation with Min vs Mean
With MSE, using `min` aggregation is a pessimistic underestimation that counters offline RL overestimation. With BSC, Q-values are probabilities — the "min" ensemble aggregation now underestimates success probability, which may be overly pessimistic in the offline setting where the dataset already represents suboptimal behavior. Consider experimenting with `mean` aggregation for BSC.

### 10.6 OGBench Dataset Format Uncertainty
The design assumes OGBench's `terminals` field marks timeout boundaries. This needs to be **verified** before deploying BSC on OGBench. If OGBench only provides `masks = 1 - success_flag` without a separate trajectory-boundary `terminals` field, the timeout case cannot be identified and BSC cannot correctly assign `target=0.0` to timeout steps.

---

## 11. Summary: Changes at a Glance

| Component | Current FQL | BSC FQL |
|-----------|------------|---------|
| Critic output | Unbounded scalar | `sigmoid(logit) ∈ [0,1]` |
| Critic loss | `MSE(Q, r + γ*mask*Q_target)` | `BCE(logit, target_bsc)` |
| TD target | `r + γ * mask * Q_target` | `(1-mask)*1 + mask*(1-truncated)*Q_target` |
| Discount γ | 0.99 | 1.0 |
| Reward used | Yes | No |
| Actor Q-loss | `-Q.mean()` (unbounded) | `-Q.mean()` (`∈ [-1, 0]`) |
| `normalize_q_loss` | False (recommended False) | **Must be False** (or re-designed) |
| `alpha` | 10 | Needs re-tuning (try 1–5) |
| Buffer changes | None | None |
| Network changes | None | Sigmoid head on `Value` |

---

## 12. Implementation Plan

### Step 1: Add `bsc` mode to `Value` network

In `utils/networks.py`, modify the `Value.__call__` method to accept a `critic_loss_type` flag (or add a `use_bsc: bool` parameter). Add a `return_logits` optional argument.

### Step 2: Add `compute_bsc_target` helper

A new function (can live inline in `critic_loss` or as a utility):

```python
def compute_bsc_target(masks, terminals, next_q):
    """BSC Bellman target with gamma=1.0.
    masks=0 -> success -> 1.0
    terminals*masks=1 -> timeout -> 0.0
    else -> bootstrap from next_q
    """
    truncated = terminals * masks
    return (1.0 - masks) * 1.0 + masks * (1.0 - truncated) * next_q
```

### Step 3: Modify `critic_loss` in `fql.py`

Branch on `self.config['critic_loss_type']`:
- `'mse'` (default): existing code unchanged
- `'bce'`: use `compute_bsc_target` + `optax.sigmoid_binary_cross_entropy`

### Step 4: Add new config keys

```python
# In get_config():
critic_loss_type='mse',  # 'mse' or 'bce'
```

### Step 5: New training script

```bash
# runscripts/fql/train_antmaze_large_bsc.sh
python main.py \
    --env_name=antmaze-large-navigate-singletask-v0 \
    --agent=agents/fql.py \
    --agent.alpha=3 \           # re-tune from 10
    --agent.discount=1.0 \      # required for pure P(success)
    --agent.critic_loss_type=bce \
    --save_dir=./exp
```

---

## 13. Experiments to Validate

1. **Sanity check**: Q-values should be in [0,1] and correlate with success rate. Log `q` mean and variance during training.
2. **Ablation γ**: Compare γ=1.0 vs γ=0.99 with BSC.
3. **Ablation label smoothing**: Compare `target ∈ {0, 1}` vs `target ∈ {0.05, 0.95}`.
4. **Alpha sweep**: Try `alpha ∈ {0.5, 1, 3, 5, 10}`.
5. **Aggregation**: Compare `q_agg='min'` vs `q_agg='mean'` with BSC.
6. **Baseline comparison**: FQL (MSE, γ=0.99) vs BSC-FQL (BCE, γ=1.0) on AntMaze large.
