

What is the connection and differences between the following
Adaptive Contrastive Masked Autoencoders for Structured Representation Learning (https://cs231n.stanford.edu/2025/papers/CS231N%20-%20ACMAE%20-%20Karan%20Singh.pdf)
Contrastive Masked Autoencoders (CME) (https://arxiv.org/abs/2207.13532) 
Visual Contrastive Masked AutoEncoder (ViC-MAE) (https://arxiv.org/abs/2303.12001)
ViC-MAE = Masked AutoEncoders (MAE) + contrastive learning.
VAEs
## Taxonomy & Relationships

```
Variational Autoencoder (VAE)
  └── Generative model (separate lineage)

Masked Autoencoder (MAE)  ←── foundation
  ├── CMAE    (+self-supervised contrastive, dual-branch architecture)
  ├── ViC-MAE (+self-supervised contrastive, extends to video)
  └── ACMAE   (+supervised contrastive with coarse labels, adaptive masking/weighting)
```

---

## Individual Descriptions

### MAE (He et al., CVPR 2022)
- **Task:** Self-supervised pretraining for ViTs
- **Method:** Mask 75% of patches → encoder sees only visible 25% → decoder reconstructs masked pixels
- **Loss:** MSE on masked patch pixels
- **Latent:** No explicit structure imposed; features emerge from reconstruction
- **Limitation:** Reconstruction alone doesn't guarantee the [CLS] token is semantically discriminative

### CMAE (Huang et al., TPAMI 2023)
- **Builds on:** MAE
- **Key addition:** Dual-branch architecture — (1) reconstruction branch (standard MAE) + (2) contrastive branch with a momentum encoder (MoCo-style)
- **Insight:** The masked view (25% patches) vs. full view naturally forms two "augmented views" for contrastive learning — no explicit crop/color augmentation needed
- **Loss:** `L = L_reconstruct + λ * L_contrastive` (InfoNCE between masked-encoder CLS and momentum-encoder CLS)
- **Supervision:** Fully self-supervised (no labels)
- **Result:** CLS token becomes both reconstructive AND instance-discriminative

### ViC-MAE (Hernandez et al., 2023)
- **Builds on:** MAE + contrastive (same family as CMAE)
- **Key addition:** Extends to **video** (temporal contrastive across frames) + images
- **Method:** MAE reconstruction + contrastive loss between encoder CLS and a momentum target
- **Supervision:** Self-supervised
- **Difference from CMAE:** (1) video-capable, (2) simpler single-branch design (no separate momentum encoder architecture — uses EMA of the same encoder), (3) different contrastive formulation (VICReg-style variance/invariance/covariance vs. InfoNCE)

### ACMAE (Singh, CS231n 2025)
- **Builds on:** CMAE / ViC-MAE framework
- **Key addition:** Injects **coarse class labels** into the contrastive loss (SupCon instead of self-supervised InfoNCE) + **adaptive** masking ratio or loss weighting based on sample difficulty
- **Supervision:** Semi-supervised (uses category labels)
- **Insight:** Coarse labels provide "free" semantic structure — pulling same-class representations together, pushing different classes apart — which pure self-supervised contrastive cannot guarantee
- **Result:** Latent space has explicit class-level clustering, better for low-data downstream tasks

### VAE (Kingma & Welling, 2014)
- **Purpose:** Generative model (sample new data)
- **Method:** Encoder maps `x → q(z|x)` (Gaussian parameters μ, σ²); decoder maps `z → p(x|z)`
- **Loss:** `L = L_reconstruct + D_KL(q(z|x) || p(z))` where p(z) = N(0, I)
- **Key property:** KL regularization forces smooth, continuous latent space → can sample z ~ N(0,I) and decode to generate new images
- **NOT masked:** Sees full input, no patch masking

---

## Differences at a Glance

| | MAE | CMAE | ViC-MAE | ACMAE | VAE |
|---|---|---|---|---|---|
| **Goal** | Representation | Representation | Representation | Representation | Generation |
| **Masking** | Yes (75%) | Yes | Yes | Yes (adaptive) | No |
| **Reconstruction loss** | MSE pixels | MSE pixels | MSE pixels | MSE pixels | MSE or BCE |
| **Contrastive loss** | ✗ | ✓ (self-sup) | ✓ (self-sup) | ✓ (supervised) | ✗ |
| **Labels needed** | No | No | No | Yes (coarse) | No |
| **Domain** | Image | Image | Image + Video | Image | Image |
| **Can generate?** | No | No | No | No | Yes |
| **Latent regularization** | None | Implicit (contrastive) | Implicit (contrastive) | Explicit (SupCon) | KL to N(0,I) |
| **Architecture** | ViT enc + shallow dec | Dual-branch + momentum enc | Single-branch + EMA target | Dual-branch + adaptive | Enc + Dec (any arch) |

---

## Key Conceptual Connections

1. **MAE → CMAE/ViC-MAE:** "Reconstruction alone gives good features, but adding contrastive alignment makes the CLS token linearly separable without fine-tuning."

2. **CMAE → ACMAE:** "Self-supervised contrastive groups by visual similarity; supervised contrastive groups by semantic category. When coarse labels are available, use them."

3. **All MAE-family vs. VAE:** The word "autoencoder" is misleading. MAE-family are **not** generative — they use encode-decode as a **pretext task** for learning representations. VAEs use encode-decode as a **generative framework** with probabilistic latent sampling. They serve completely different purposes.

4. **In our project:** The SD-VAE (`stabilityai/sd-vae-ft-mse`) used by DiT is a true VAE — it compresses images to latent space for diffusion. The CMAE/ACMAE encoder is used as a **feature extractor** for conditioning. These are two different models doing two different jobs.

---

## Why This Matters for the Project

The CMAE/ACMAE encoder provides **discriminative image features** for conditioning the DiT. The progression MAE → CMAE → ACMAE adds increasingly strong semantic structure to the features:

- **MAE features:** good general representations, but CLS token may not cleanly separate classes
- **CMAE features:** instance-discriminative CLS token (knows "this is different from that")
- **ACMAE features:** class-discriminative CLS token (knows "this is the same category as that")

For conditioning a generative model to produce class-consistent image variations, **ACMAE's class-structured features should provide the strongest conditioning signal** — which is what the proposal leverages.

ViT (Vision Transformer) is not a sibling — it's the **underlying backbone architecture** that MAE and its descendants are built on top of. Think of it as a layer below:

```
ViT (Vision Transformer) ←── backbone architecture (Dosovitskiy et al., 2020)
  │
  ├── Supervised ViT        (train ViT with labeled data, standard classification)
  │
  ├── MAE                   (self-supervised PRETRAINING METHOD for ViT)
  │     ├── CMAE
  │     ├── ViC-MAE
  │     └── ACMAE
  │
  └── DINO / DINOv2         (another self-supervised method for ViT, for reference)

VAE ←── separate model family (can use any architecture, not ViT-specific)
```

**The distinction:**

- **ViT** = a neural network architecture (patch embedding → transformer encoder → output). It defines *what the model looks like*.
- **MAE/CMAE/ViC-MAE/ACMAE** = training methods (how to train a ViT without labels). They define *how the model learns*.

MAE's encoder **is** a ViT. When you load `facebook/vit-mae-base`, you get a ViT-B/16 that was *trained using the MAE method*. When you load `google/vit-base-patch16-224`, you get the same ViT-B/16 architecture but *trained with supervised ImageNet labels*.

Same architecture, different training recipe — different feature quality.

DiT (Diffusion Transformer) sits in the **generative** branch, not the representation learning branch. Full picture:

```
ViT (Vision Transformer) ←── backbone architecture
  │
  ├─── REPRESENTATION LEARNING (discriminative)
  │     │
  │     ├── Supervised ViT       (labeled classification)
  │     ├── DINO / DINOv2        (self-supervised, no masking)
  │     └── MAE                  (self-supervised, mask+reconstruct)
  │           ├── CMAE
  │           ├── ViC-MAE
  │           └── ACMAE
  │
  └─── GENERATION (generative)
        │
        ├── DiT                  (ViT as the denoising backbone in latent diffusion)
        └── PixArt, SD3, etc.   (same idea, scaled up)

VAE ←── separate generative family
  │
  └── SD-VAE                    (used BY DiT to compress images to latent space)

Diffusion Process ←── separate framework
  │
  ├── U-Net backbone             (Stable Diffusion 1/2, DDPM)
  └── ViT/DiT backbone           (DiT, SD3) ← replaces U-Net with a transformer
```

**How they connect in our project:**

```
Reference Image
      │
      ▼
  CMAE/ACMAE encoder (representation branch) ──→ conditioning vector
      │                                                │
      │                                                ▼
  SD-VAE encoder ──→ latent z ──→ DiT (generation branch) ──→ denoised latent ──→ SD-VAE decoder ──→ output image
```

- **CMAE/ACMAE** = representation learning side (trained with MAE method, produces features)
- **DiT** = generation side (trained with diffusion, produces images)
- **SD-VAE** = utility (compresses images to/from latent space for DiT to work in)

DiT and MAE both use ViT as their architecture, but they solve opposite problems: MAE learns to *understand* images, DiT learns to *create* them. The project bridges the two by using MAE-family features to *tell* DiT *what* to create.

Mostly right, but ACMAE and ViC-MAE descriptions are swapped/off. Corrected:

| Model | One-line intuition | What changed |
|-------|-------------------|--------------|
| **ViT** | "How do we process images with a transformer?" | ✅ Correct |
| **MAE** | "Learn by filling in missing parts." | ✅ Correct |
| **CMAE** | "Also learn what images are similar." | ✅ Correct (adds self-supervised contrastive) |
| **ViC-MAE** | "Also learn what images are similar, **and do it for video too** — keep representations diverse and stable via variance/covariance regularization." | Your ACMAE description ("diverse and stable") actually fits here — ViC-MAE uses VICReg-style loss which explicitly enforces variance (diverse) and decorrelation (stable) |
| **ACMAE** | "Use **class labels** to learn which images belong together, and **adaptively** focus masking/weighting on harder samples." | Your description ("pay attention to the right parts") is closer but the key word is **labels** — ACMAE is the only one that uses supervised contrastive, not self-supervised |

**The progression in plain English:**

```
MAE:     "Guess the missing pixels"
CMAE:    "Guess the missing pixels AND pull similar images closer in feature space"
ViC-MAE: "Same as CMAE, but also works on video and prevents feature collapse"
ACMAE:   "Same as CMAE, but I'll TELL you which images are similar (via class labels)
           and spend more effort on the hard ones"
```

The critical distinction: MAE → CMAE → ViC-MAE are all **self-supervised** (no labels). ACMAE breaks that by injecting **coarse labels** — which is why it produces the most class-structured features, and why it's the strongest choice for conditioning a generative model on semantic category.

## Complete Summary

### ViT — Vision Transformer (Dosovitskiy et al., 2020)
**What:** A neural network architecture that applies the transformer (originally designed for text) to images. Splits an image into fixed-size patches (e.g., 16×16), linearly embeds each patch into a token, adds positional embeddings, and processes them through standard transformer encoder layers (self-attention + MLP).

**Role:** The **backbone architecture** underneath everything else below. Not a training method — just a model structure. Can be trained with any objective (supervised, self-supervised, generative).

---

### VAE — Variational Autoencoder (Kingma & Welling, 2014)
**What:** A generative model. Encoder compresses input `x` into a probability distribution `q(z|x)` (mean + variance). A latent `z` is sampled from this distribution. Decoder reconstructs `x` from `z`. Trained with reconstruction loss + KL divergence that forces the latent space toward a smooth Gaussian N(0, I).

**Key property:** The KL regularization makes the latent space continuous and smooth — you can sample any `z ~ N(0, I)` and decode it into a plausible image. This is what makes it generative.

**In our project:** The SD-VAE (`stabilityai/sd-vae-ft-mse`) compresses 256×256 images into 32×32 latent maps. DiT operates entirely in this latent space (cheaper than pixel space).

---

### MAE — Masked Autoencoder (He et al., CVPR 2022)
**What:** A self-supervised pretraining method for ViT. Randomly masks 75% of image patches. The encoder (a ViT) sees only the visible 25%. A lightweight decoder reconstructs the masked patches' pixels. Loss = MSE on masked pixels only.

**Key insight:** By forcing the encoder to reconstruct from very sparse input, it must learn rich internal representations of image structure. No labels needed.

**Limitation:** The [CLS] token is optimized for reconstruction, not discrimination — two images of different classes might have similar CLS features if they share low-level texture.

---

### CMAE — Contrastive Masked Autoencoder (Huang et al., TPAMI 2023)
**What:** MAE + self-supervised contrastive learning. Dual-branch architecture: (1) online encoder processes masked patches (like MAE), (2) momentum encoder (MoCo-style, EMA-updated) processes the full unmasked image. An InfoNCE contrastive loss pulls the masked-view CLS token toward the full-view CLS token of the same image, and pushes it away from other images' tokens.

**Loss:** `L = L_reconstruct + λ * L_contrastive`

**Key insight:** Masking naturally creates two "views" of the same image (masked vs. full) — perfect for contrastive learning without needing crop/color augmentations.

**Improvement over MAE:** CLS token becomes instance-discriminative (can tell images apart), not just reconstructive.

---

### ViC-MAE — Visual Contrastive Masked Autoencoder (Hernandez et al., 2023)
**What:** MAE + contrastive learning, extended to **images and video**. Uses a VICReg-style loss (variance + invariance + covariance regularization) instead of InfoNCE. Single-branch with EMA target (simpler than CMAE's dual-branch).

**Loss:** `L = L_reconstruct + λ * L_VICReg`

**Key differences from CMAE:**
- Works on video (temporal contrastive across frames)
- VICReg loss explicitly prevents feature collapse (variance term) and decorrelates feature dimensions (covariance term), making representations more stable
- Simpler architecture (no separate momentum encoder branch)

**Improvement over MAE:** Same as CMAE (discriminative CLS token) + video capability + explicit anti-collapse guarantees.

---

### ACMAE — Adaptive Contrastive Masked Autoencoder (Singh, CS231n 2025)
**What:** CMAE/ViC-MAE framework + **supervised contrastive loss using coarse class labels** + adaptive masking or loss weighting. Instead of self-supervised contrastive (same image = positive, different image = negative), uses SupConLoss: same-class images = positives, different-class = negatives. Adapts difficulty by focusing more on hard samples.

**Loss:** `L = L_reconstruct + λ * L_SupCon(labels)`

**Key difference from CMAE/ViC-MAE:** Uses **labels**. This is the only semi-supervised method in the MAE family. The labels don't need to be fine-grained — coarse category labels suffice.

**Improvement:** Latent space has explicit class-level clustering. Two images of "dog" are pulled together even if they look very different. Strongest semantic structure of all MAE variants.

---

### DiT — Diffusion Transformer (Peebles & Xie, ICCV 2023)
**What:** A generative model that replaces the U-Net backbone in latent diffusion with a ViT. Operates in VAE latent space: noisy latent `z_t` is patchified into tokens, processed by transformer blocks with adaptive layer norm (adaLN-Zero) that injects the diffusion timestep `t` and conditioning signal `c` (e.g., class label). Trained to predict the noise `ε` added at each timestep.

**Generation process:** Sample pure noise `z_T ~ N(0, I)` → iteratively denoise `z_T → z_{T-1} → ... → z_0` using the trained DiT → decode `z_0` through SD-VAE → output image.

**Key insight:** Transformers scale better than U-Nets — larger DiT = better FID, following a clean scaling law.

**In our project:** The generation backbone. We train DiT-S/2 (33M params) to generate images, conditioned on ACMAE features instead of class labels, with error-aware training on the denoising chain.

---

## How They All Fit Together

```
                    UNDERSTANDING images              CREATING images
                    (representation)                  (generation)
                          │                                │
    ┌─────────────────────┤                    ┌───────────┤
    │                     │                    │           │
   MAE                  CMAE               SD-VAE        DiT
 (reconstruct)    (+ contrastive)        (compress)   (denoise)
    │                     │                    │           │
 ViC-MAE              ACMAE                   └─────┬─────┘
 (+ video)          (+ labels)                      │
                                              Latent Diffusion
    └────────── all use ViT architecture ──────────┘
```

```
Reference Image → ACMAE encoder (understand) → features → DiT (create) → SD-VAE decode → New Image
```


































