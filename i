Project Proposal: Error-Aware Contrastive-Conditioned Diffusion Transformers for Image Variation Generation
1-Sentence Overall Idea
Investigate whether combining error-aware training (adapted from Matrix Game 3.0's residual-injection self-correction) with contrastive-MAE-pretrained image encoders (replacing class-label conditioning) measurably improves a Diffusion Transformer's image variation quality, measured by FID and CLIP score on an ImageNet subset.
Background
Diffusion Transformers (DiT; Peebles & Xie, 2023) established that replacing the U-Net backbone in latent diffusion with a Vision Transformer yields state-of-the-art class-conditional image generation on ImageNet. Standard DiT conditions on class labels via an embedding lookup fed through adaptive layer norm (adaLN-Zero). Two recent lines of work suggest complementary improvements:

Error-Aware Training (Matrix Game 3.0, Wang et al. 2025): In autoregressive diffusion video generation, the model sees only clean ground-truth frames during training but must condition on its own imperfect predictions at inference, causing error accumulation. Matrix Game 3.0 addresses this by storing the model's prediction residuals (R = x_hat - x) in an error buffer and re-injecting them into training inputs: x_tilde = x + alpha * R. The model then must still predict correct outputs given corrupted inputs, learning self-correction. This is conceptually related to Scheduled Sampling (Bengio et al., 2015) but uses the model's own structured errors rather than random corruption -- making it more realistic. This technique has not been studied for image-conditioned DiTs.
Justification: “If the model will see imperfect inputs at inference time, train it on imperfect inputs too”
Contrastive Masked Autoencoders (CMAE/ViC-MAE): Standard MAE pretraining (He et al., 2022) learns features via pixel reconstruction but provides no explicit incentive for the latent space to be semantically discriminative. CMAE (Huang et al., 2023) and ViC-MAE (Hernandez et al., 2023) add a supervised contrastive loss on the encoder's [CLS] token during MAE pretraining, producing features that are both reconstructive and discriminative. The ACMAE framework (Singh, CS231n 2025) further demonstrates that injecting coarse labels via contrastive alignment during MAE pretraining improves downstream tasks in low-data regimes.
Justification: Contrastive methods build distinct embeddings by aligning similar views or labels, which results in features that are easier to separate and better suited for specific tasks. While masked autoencoding focuses on reconstructing fine-grained local details, contrastive learning applies a global constraint to the overall geometry of the feature space.
Conditional Signal

This project fuses both ideas into an image-to-image DiT: (a) replace class-label conditioning with CMAE image-feature conditioning (given a reference image, generate a new variation), and (b) apply error-aware training so the model learns to generate high-quality outputs even when conditioning features are imperfect. The architecture is a standard DiT-S/2 where the class-embedding lookup is replaced by a frozen CMAE encoder + linear projection producing the adaLN conditioning vector.
Falsifiable Hypothesis
A DiT-S/2 (33M parameters) trained for 100K steps on ImageNette (10-class ImageNet subset, 256x256) with error-aware conditioning injection (alpha=0.3, FIFO buffer of size 512, injection probability p=0.5) will achieve FID-5K at least 10% lower (better) than the identical model trained without error-aware injection. Independently, replacing class-label conditioning with CMAE-pretrained ViT-B/16 image features will reduce FID-5K by at least 5%. The combination will yield a total FID reduction of at most 12% (partially redundant gains).

Why this is non-trivial and falsifiable:

Error-aware injection could degrade FID if model residuals act as destructive noise rather than useful augmentation (the structured-error hypothesis could be wrong for images vs. video).
CMAE features could underperform simple class embeddings if the 768-dim CLS token provides less class-separable conditioning than a learned 10-class embedding (information bottleneck).
The predicted interaction effect (partial redundancy) could go either direction -- the gains could be additive or fully redundant.
Metrics
FID-5K (primary): Frechet Inception Distance over 5,000 generated samples vs. validation set. Lower is better. Measures joint quality + diversity.
Inception Score (secondary): Over the same 5,000 samples. Higher is better. Measures per-sample quality and class separability.
CLIP Score (auxiliary): Cosine similarity between CLIP embeddings of the reference image and the generated image. Measures semantic fidelity of the image-to-image mapping.
LPIPS (auxiliary): Learned Perceptual Image Patch Similarity between reference and generated image. Ensures the model produces variations, not copies (moderate LPIPS is desired).
Statistical test: Mean +/- std over 3 random seeds; paired t-test (p < 0.05) for primary FID comparisons.
Methodology
Innovation 1: Error-Aware Training for DiT (adapted from Matrix Game 3.0)
During training, the model's own denoising errors are collected and re-injected into the conditioning signal:

Error Buffer: Maintain a FIFO buffer B of size 512, storing latent-space prediction residuals.
Residual Collection: After each training step, run a single-step DDIM decode of the predicted noise to obtain predicted clean latent z_hat. Compute R = z_hat - z_target. Append R to buffer (buffer begins filling after a 1,000-step warmup).
Error Injection: With probability p=0.5, sample R_i from B and corrupt the conditioning vector: c_tilde = c + alpha * proj(R_i), where proj() is a frozen random linear map from latent dim to conditioning dim (initialized once, not trained).
Training: The model must still predict the correct noise epsilon given corrupted conditioning, learning self-correction against its own failure modes.
Ablation: Sweep alpha in {0.1, 0.3, 0.5}.
Innovation 2: CMAE-Pretrained Image Conditioning
Replace the class-label embedding with image-derived features from a contrastive MAE encoder:

Initialize from facebook/vit-mae-base (ViT-B/16, 86M params, MAE-pretrained on ImageNet-1K).
Add contrastive head: MLP projector (768 -> 256) on the [CLS] token. Apply SupConLoss using the 10 ImageNette class labels as coarse supervision.
Fine-tune the full encoder + contrastive head for 20 epochs on ImageNette (combined loss: L = L_reconstruct + 0.1 * L_contrastive).
Freeze the CMAE encoder after pretraining.
Conditioning: Extract [CLS] token (768-dim) from a reference image. Linear projection (768 -> 384) maps it to DiT-S/2's conditioning dimension, replacing the class embedding lookup in adaLN-Zero.
At training time, the reference image is a randomly sampled same-class image (not the target), ensuring the model generalizes rather than memorizes.
Experimental Design
Step
Task
Time (T4)
Time (H100)
1
Data preparation: Download ImageNette 320px via HuggingFace. Resize to 256x256. Pre-extract VAE latents using stabilityai/sd-vae-ft-mse for all train/val images.
0.5 h
0.1 h
2
CMAE encoder fine-tuning: Init from facebook/vit-mae-base. Add contrastive head. Train 20 epochs on ImageNette, batch 64, lr=1e-4, AdamW.
2 h
0.3 h
3
Exp A -- Baseline DiT-S/2: Train from scratch on ImageNette latents, class-conditional, 100K steps, batch 128, lr=1e-4, fp16 + gradient checkpointing.
7 h
1 h
4
Exp B -- Error-Aware DiT-S/2: Same as Step 3 + error-aware training loop (alpha=0.3, buffer 512, p=0.5).
8 h
1.2 h
5
Exp C -- CMAE-Conditioned DiT-S/2: Same as Step 3 but replacing class embedding with frozen CMAE encoder features.
8 h
1.2 h
6
Exp D -- Combined: Both innovations together. Same hyperparams as B and C.
9 h
1.3 h
7
Evaluation: Generate 5K images per model. Compute FID-5K, IS, CLIP Score, LPIPS. 3 seeds per config.
4 h
0.5 h
8
Ablation on alpha: Error-aware training with alpha in {0.1, 0.3, 0.5} using best conditioning from Steps 3-6.
10 h
1.5 h
9
Gaussian-noise control: Same as Step 4 but inject random Gaussian noise of matched norm instead of model residuals.
8 h
1.2 h

Baseline
B1 -- Standard DiT-S/2 (primary control): Class-conditional DiT-S/2 trained with standard training on ImageNette. No error-aware injection, no CMAE conditioning.
B2 -- Standard-MAE-Conditioned DiT-S/2: Same as Exp C but using facebook/vit-mae-base without the contrastive fine-tuning step.
B3 -- Gaussian-Noise-Injected DiT-S/2: Same as Exp B but replacing structured model residuals with random Gaussian noise of matched L2 norm.
Resources/Assets
Models
Model
ID / Source
Params
Disk
MAE ViT-B/16 (encoder init)
facebook/vit-mae-base
86M
~330 MB
Stable Diffusion VAE
stabilityai/sd-vae-ft-mse
83M
~335 MB
DiT-S/2 (trained from scratch)
via chuanyangjin/fast-DiT code
33M
~130 MB

Datasets
Dataset
HuggingFace ID
Size
Details
ImageNette
frgfm/imagenette
~1.5 GB
10-class ImageNet subset, 320px

Code
fast-DiT: https://github.com/chuanyangjin/fast-DiT (Optimized single-GPU training)
facebookresearch/DiT: https://github.com/facebookresearch/DiT (Model architecture reference)
facebookresearch/mae: https://github.com/facebookresearch/mae (MAE definitions)
ViC-MAE: https://github.com/jeffhernandez1995/ViC-MAE (Contrastive reference)
CMAE: https://github.com/ZhichengHuang/CMAE (Contrastive head implementation)
Compute Estimation
Phase
T4 (16 GB)
L4 (24 GB)
H100 (80 GB)
VAE extraction + CMAE fine-tune
2.5 h
1.8 h
0.4 h
DiT-S/2 x4 factorial configs
32 h
22 h
4.7 h
Evaluation & Metrics
4 h
3 h
0.5 h
Ablations & Controls
18.5 h
13 h
2.7 h
Total Experiment Suite
~57 h
~40 h
~8.3 h


VRAM budget (peak): DiT-S/2 training fits within 7-8.5 GB, well within the 16 GB limit of a T4 GPU.
Papers
Peebles & Xie, "Scalable Diffusion Models with Transformers," ICCV 2023.
Wang et al., "Matrix-Game 3.0," 2025.
He et al., "Masked Autoencoders Are Scalable Vision Learners," CVPR 2022.
Huang et al., "Contrastive Masked Autoencoders are Stronger Vision Learners," IEEE TPAMI 2023.
Singh, "Adaptive Contrastive Masked Autoencoders," CS231n 2025.
