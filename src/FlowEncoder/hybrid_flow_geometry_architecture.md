# Hybrid Flow Context + Geometry Encoding Architecture

## Motivation

- Goal: build a general CFD digital twin across datasets (channels, pipes, cylinders, boundary layers, etc.).
- Challenges:
  - **Flow metadata** (Reynolds numbers, boundary conditions, flow type) vary by dataset.
  - **Geometry representations** (cylinder surfaces, channel walls, arbitrary meshes) differ.
- Solution: **Decouple flow physics from geometry** via two encoders:
  1. **FlowTypeEncoder** – encodes flow regime, Reynolds number, and BCs.
  2. **Geometry Encoder** – encodes spatial coordinates or wall-node embeddings.
- The two are fused into a shared latent representation for downstream prediction of surface quantities (e.g., WSS, TAWSS, velocity, pressure).

---

## FlowTypeEncoder

- **Inputs**
  - **Continuous Scalars (normalized, masked)**: log(Re), free-stream turbulence (FST), roughness, etc. (mask indicates missing/known).
  - **Categorical Attributes**: geometry (channel, pipe, artery, …), drive type, development, wall curvature, periodicity, wall state, dimensionality.&#x20;
- **Architecture**: scalars + embeddings → concatenation → 2-layer MLP (~64 hidden units) → latent vector **c ∈ ℝ^32**.

---

## Geometry Encoder

- Maps positions/surface coordinates (x, y, z) into embeddings.
- Options: Use our previous MPNN+UNET architecture
- Produces per-node features, g(x).

---

## Fusion

- Two main approaches:
  1. **Concatenation**: broadcast flow latent **c** across nodes, concatenate with g(x).
  2. **FiLM (Feature-wise Linear Modulation)**: map c -> (γ, β), apply to g(x): g′ = γ*g + β.
- Result passed to prediction head.

---

## Task Parameterization

- Add a task embedding e_task for flexible outputs.
- Tasks: pressure, velocity, WSS, TAWSS, Cf, etc.
- Options:
  - Append e_task to [g, c].
  - Use e_task to generate task-specific head.

---

## Training Setup

- **Targets**: WSS, TAWSS, pressure, mean velocity profiles.
- **Loss**: Huber or L1, with uncertainty-weighted multi-task loss.
- **Regularization**:
  - Smoothness penalty on surfaces.
  - Monotonicity penalty across log(Re) where dictated by physics.

---

## Workflow

1. Preprocess dataset: extract geometry → Geometry Encoder inputs. Build flow vector → FlowTypeEncoder. Assign task ID.
2. Forward pass: compute g(x), compute c, fuse [g, c, e_task], predict field.
3. Inference: adjust Reynolds number freely (log_Re knob), add new flow types by specifying categorical attributes (no code change).

---

## Advantages

- **Scalable**: add new flow types with metadata only.
- **Generalizable**: works for laminar/turbulent, cylinders, pipes, boundary layers.
- **Controllable**: Reynolds number is a free input.
- **Flexible**: choose output metric at runtime.

---

# Our Prototype so far (simple model)

## Inputs (Context)

- log_10(Reynolds-like number).
- One-hot of **family** (dataset/geometry label).
- One-hot of **Re-kind** (τ, θ, or plain Re).

## FlowEncoder

- 2-layer MLP: `[Linear(in_dim → 64) → ReLU → Linear(64 → 32)]`.
- Produces a latent **c ∈ ℝ³²**.

## Fusion

- Concatenation of **[c, g]** if geometry is used.
- Otherwise just **c**.
- Output goes through a **Projector MLP** -> 128-d normalized vector.

## Training

- **Objective**: contrastive InfoNCE.
  - Positives: embeddings from same family.
  - Negatives: embeddings from different families.

---

Our prototype is a minimal working skeleton of the FlowType+Geometry concept, but missing supervised physics heads, task parameterization, and the more sophisticated geometry handling. We will continue to work on this in CS494 and with our mentor.
