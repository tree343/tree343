Great question. Let me work through how the ACMAE contrastive framework translates from static images with class labels to video frames with temporal structure.

## The Core Problem

ACMAE uses **class labels** as the grouping signal for SupConLoss:
- Same class → positive pair (pull together)
- Different class → negative pair (push apart)

Video frames don't have discrete class labels. But they have something arguably better: **temporal proximity**.

## The Mapping: Classes → Temporal Neighborhoods

| ACMAE (images) | Video adaptation |
|---|---|
| Class label "dog" | Temporal window (frames within ±k of each other) |
| Same class = positive | Nearby frames = positive (frame_t, frame_{t+1}) |
| Different class = negative | Distant frames = negative (frame_t, frame_{t+100}) |
| Coarse superclass | Scene/clip boundary (frames from same scene) |

**The intuition:** Consecutive frames share semantic content (same scene, same objects, similar layout) just like same-class images share semantic content. Temporal proximity IS the "coarse label" for video.

## Concrete Formulation

### Level 1: Binary temporal contrastive (simplest)
```
Positive pairs: (frame_t, frame_{t+1})     — adjacent frames
Negative pairs: (frame_t, frame_{t+100})   — distant frames

Loss: L_reconstruct + λ * InfoNCE(positives, negatives)
```
This is essentially what ViC-MAE already does for video. Simple, no labels needed.

### Level 2: Soft temporal contrastive (better — direct ACMAE analog)
Instead of binary positive/negative, use **temporal distance as a soft label**:
```
similarity_target(frame_i, frame_j) = exp(-|i - j| / τ)

Frame_t and frame_{t+1}:   similarity ≈ 0.95  (very positive)
Frame_t and frame_{t+5}:   similarity ≈ 0.78  (somewhat positive)
Frame_t and frame_{t+50}:  similarity ≈ 0.01  (negative)
```
This is richer than ACMAE's binary class signal — it provides a **continuous hierarchy** of similarity, like having infinitely many superclasses ordered by temporal distance.

### Level 3: Scene-based grouping (closest ACMAE analog)
If the video has scene/shot boundaries (either annotated or detected via simple histogram differencing):
```
"Superclass" = scene ID
"Class" = temporal position within scene

SupConLoss: pull frames from same scene together, push different scenes apart
+ Temporal weighting: within a scene, closer frames are stronger positives
```
This directly mirrors ACMAE's two-level hierarchy:
- ACMAE: superclass (coarse) → class (fine)
- Video: scene (coarse) → temporal position (fine)

## How This Fits the Full Pipeline

```
TRAINING (video variant):

1. Sample frame_t and frame_{t+1} from a video clip
        ↓
   frame_t → [ACMAE-Video encoder (frozen)] → CLS_t (768-d) → Linear → c (384-d)
   frame_{t+1} → [SD-VAE encoder (frozen)] → z₀ (target latent)
        ↓
   [Add noise at random timestep] → zₜ
        ↓
   DiT(zₜ, timestep, c) → ε_pred
   Loss: MSE(ε_pred, ε_true)

INFERENCE:
   Given frame_t → ACMAE encoder → c → DiT denoises from noise → SD-VAE decode → predicted frame_{t+1}
```

The task is now **next-frame prediction**: given frame_t, generate frame_{t+1}. This is:
- Unambiguously image-to-image (frame in → frame out)
- Naturally autoregressive (can chain: frame_1 → frame_2 → frame_3 → ...)
- **Error-aware training is DIRECTLY motivated** — at inference, frame_{t+2} is conditioned on the model's generated frame_{t+1}, not the real one

## Why Error-Aware Training Now Makes MUCH More Sense

This is the key advantage of switching to video. The concern about residual reinjection on static images is **completely eliminated**:

| | Static image generation | Video frame prediction |
|---|---|---|
| Autoregressive? | Only in denoising chain (weak) | Yes — frame_t → frame_{t+1} → frame_{t+2} (strong) |
| Error accumulation source | Denoising steps (implicit) | Generated frame conditioning (explicit, exactly like MG3.0) |
| Train/test mismatch | Mild (clean z_t vs. model z_t) | Severe (real frame_t vs. generated frame_t at inference) |
| MG3.0 analogy | Indirect | **Direct** — same problem, same solution |

The error-aware training now works **exactly as Matrix Game 3.0 intended**:
1. Train: condition on real frame_t → predict frame_{t+1}
2. Error buffer: store R = generated_frame_{t+1} − real_frame_{t+1}
3. Error injection: corrupt frame_t features with stored residuals: `c̃ = c + α·proj(R)`
4. Model learns to predict frame_{t+1} correctly even from imperfect conditioning

## ACMAE Pretraining for Video Encoder

The encoder pretraining step adapts cleanly:

```
Standard ACMAE (images):
  - Mask 75% patches of image
  - Reconstruct masked patches
  - SupConLoss: same-class CLS tokens → pull together

Video ACMAE:
  - Mask 75% patches of frame_t
  - Reconstruct masked patches
  - Temporal SupConLoss:
      Positive: CLS(frame_t) ↔ CLS(frame_{t+1})   [adjacent = same "class"]
      Negative: CLS(frame_t) ↔ CLS(frame_{t+100})  [distant = different "class"]
      Optional soft weighting: similarity ∝ exp(-|Δt| / τ)
```

This produces an encoder whose CLS token captures **what's happening in the scene right now** — object positions, camera angle, lighting — exactly what DiT needs to predict what happens next.

## What Changes in the Proposal

| Aspect | Image version | Video version |
|---|---|---|
| **Task** | Image variation (same class) | Next-frame prediction |
| **Dataset** | ImageNette (10 classes, 9.5K images) | UCF-101 or Kinetics-400 subset (video clips) |
| **ACMAE labels** | Class integers | Temporal distance (soft) or scene IDs |
| **DiT conditioning** | CLS of random same-class image | CLS of frame_t |
| **DiT target** | Random same-class image latent | frame_{t+1} latent |
| **Error-aware motivation** | Weak (denoising chain only) | **Strong** (autoregressive frame prediction) |
| **Evaluation metrics** | FID, IS, CLIP Score | FVD, LPIPS(t→t+1), PSNR, SSIM |
| **Compute** | Similar — video frames are just images | Similar — process one frame pair at a time |

### Dataset options for video:
| Dataset | Size | Details |
|---|---|---|
| **UCF-101** | ~7 GB | 13K clips, 101 action classes, 320×240 |
| **Kinetics-400 (mini subset)** | ~10 GB | Can sample 10K clips from full dataset |
| **BAIR Robot Pushing** | ~6 GB | 256×256, simple dynamics, common benchmark for frame prediction |

BAIR Robot Pushing is the most compute-friendly (256×256, simple scenes, well-benchmarked for frame prediction). UCF-101 gives richer content and has actual class labels you could use as scenes.

## Summary

The video formulation is a **strictly better fit** for the project's two innovations:
- **ACMAE:** Temporal proximity replaces class labels as the contrastive grouping signal — works naturally, no annotation needed
- **Error-aware training:** Frame-to-frame autoregression gives the **exact same** error accumulation problem that Matrix Game 3.0 was designed to solve

The only tradeoff is slightly more complex data loading (video clips instead of single images). Everything else — model architecture, training loop, compute requirements — stays essentially the same since you process one frame pair at a time.
