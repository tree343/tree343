https://www.dropbox.com/scl/fi/tznygrqla3ucpzgjr2kyk/project.md?rlkey=7h3agpfqi0y5vl1u8f8hnb1gg&st=1bpz01tl&dl=0


<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>DriveWAM Architecture Diagram</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    min-height: 100vh;
    padding: 32px;
  }
  h1 {
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 8px;
  }
  .subtitle {
    text-align: center;
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 36px;
  }
  .diagram {
    max-width: 1100px;
    margin: 0 auto;
    position: relative;
  }

  /* ---- shared box styles ---- */
  .box {
    border-radius: 12px;
    padding: 14px 18px;
    position: relative;
    text-align: center;
    font-size: 0.82rem;
    line-height: 1.45;
    box-shadow: 0 2px 12px rgba(0,0,0,.35);
  }
  .box .label {
    font-weight: 700;
    font-size: 0.92rem;
    margin-bottom: 4px;
  }
  .box .detail { color: #cbd5e1; font-size: 0.78rem; }

  /* colours per category */
  .input   { background: #1e3a5f; border: 1.5px solid #3b82f6; }
  .module  { background: #312e81; border: 1.5px solid #818cf8; }
  .process { background: #164e63; border: 1.5px solid #22d3ee; }
  .output  { background: #14532d; border: 1.5px solid #4ade80; }
  .loss    { background: #4c1d2e; border: 1.5px solid #f472b6; }

  /* grid layout ------------------------------------------------ */
  .row {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }
  .row .box { flex: 0 1 220px; }

  /* arrows (pure CSS) */
  .arrow-down {
    display: flex;
    justify-content: center;
    margin: 4px 0;
  }
  .arrow-down span {
    display: block;
    width: 2px;
    height: 28px;
    background: #475569;
    position: relative;
  }
  .arrow-down span::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 50%;
    transform: translateX(-50%);
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 8px solid #475569;
  }
  .arrow-label {
    text-align: center;
    font-size: 0.7rem;
    color: #64748b;
    margin: -2px 0 2px;
  }

  /* section titles */
  .section-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin: 18px 0 8px;
    padding-left: 4px;
  }
  .section-title.blue   { color: #60a5fa; }
  .section-title.violet { color: #a78bfa; }
  .section-title.cyan   { color: #22d3ee; }
  .section-title.green  { color: #4ade80; }
  .section-title.pink   { color: #f472b6; }

  /* legend */
  .legend {
    display: flex;
    gap: 20px;
    justify-content: center;
    margin-top: 32px;
    flex-wrap: wrap;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    color: #94a3b8;
  }
  .legend-swatch {
    width: 14px; height: 14px;
    border-radius: 4px;
  }

  /* divider */
  .phase-divider {
    border: none;
    border-top: 1px dashed #334155;
    margin: 28px 0 18px;
  }
  .phase-label {
    text-align: center;
    font-size: 0.8rem;
    font-weight: 700;
    color: #94a3b8;
    margin-bottom: 10px;
    letter-spacing: 1px;
  }

  /* wide box */
  .wide { flex: 0 1 480px !important; }
</style>
</head>
<body>

<h1>DriveWAM — Architecture Diagram</h1>
<p class="subtitle">Diffusion Transformer World-Action Model for AV Scene Prediction &amp; Closed-Loop Planning</p>

<div class="diagram">

  <!-- ==================== TRAINING ==================== -->
  <div class="phase-label">⬛ TRAINING PHASE</div>

  <div class="section-title blue">Inputs</div>
  <div class="row">
    <div class="box input">
      <div class="label">Front-Camera Frame (t)</div>
      <div class="detail">128×128 RGB image from nuScenes mini</div>
    </div>
    <div class="box input">
      <div class="label">Ego-Action</div>
      <div class="detail">Steering angle + acceleration (16-d learned embedding)</div>
    </div>
    <div class="box input">
      <div class="label">Ground-Truth Frames (t+1 … t+4)</div>
      <div class="detail">Next 4 real camera frames for diffusion loss</div>
    </div>
  </div>

  <div class="arrow-down"><span></span></div>

  <div class="section-title violet">Model Modules</div>
  <div class="row">
    <div class="box module">
      <div class="label">① ViT-S Frame Encoder</div>
      <div class="detail">Embeds frame → patch tokens; action embedding concatenated to tokens</div>
    </div>
  </div>

  <div class="arrow-down"><span></span></div>
  <div class="arrow-label">encoded frame_t + action_embedding</div>

  <div class="row">
    <div class="box module">
      <div class="label">② 4-Layer DiT (Diffusion Transformer)</div>
      <div class="detail">Denoises to predict frames t+1 … t+4 conditioned on (frame_t, action_embedding)</div>
    </div>
  </div>

  <div class="arrow-down"><span></span></div>
  <div class="arrow-label">predicted future frames + encoded representations</div>

  <div class="row">
    <div class="box module">
      <div class="label">③ Inverse Dynamics Head (2-layer MLP)</div>
      <div class="detail">Maps (encoded_frame_t, encoded_frame_{t+4}) → predicted ego-action</div>
    </div>
  </div>

  <div class="arrow-down"><span></span></div>

  <div class="section-title pink">Training Losses</div>
  <div class="row">
    <div class="box loss">
      <div class="label">DDPM Diffusion Loss</div>
      <div class="detail">Reconstruction of GT frames t+1…t+4</div>
    </div>
    <div class="box loss">
      <div class="label">Inverse Dynamics Loss</div>
      <div class="detail">Self-supervised consistency: predicted action ≈ true action</div>
    </div>
    <div class="box loss">
      <div class="label">IQL Value Loss</div>
      <div class="detail">Offline RL on logged human trajectories (separate training)</div>
    </div>
  </div>

  <!-- ==================== INFERENCE ==================== -->
  <hr class="phase-divider" />
  <div class="phase-label">▶ INFERENCE / CLOSED-LOOP PLANNING</div>

  <div class="section-title blue">Inputs</div>
  <div class="row">
    <div class="box input">
      <div class="label">Real Observed Frame (t)</div>
      <div class="detail">Live dashcam image</div>
    </div>
  </div>

  <div class="arrow-down"><span></span></div>

  <div class="section-title cyan">Planning Process</div>
  <div class="row">
    <div class="box process wide">
      <div class="label">1 — Propose K=3 Actions</div>
      <div class="detail">Generate 3 candidate steering / throttle pairs</div>
    </div>
  </div>

  <div class="arrow-down"><span></span></div>

  <div class="row">
    <div class="box process wide">
      <div class="label">2 — Imagine Futures (DiT)</div>
      <div class="detail">For each action, run the DiT to produce 3 predicted future rollouts</div>
    </div>
  </div>

  <div class="arrow-down"><span></span></div>

  <div class="row">
    <div class="box process wide">
      <div class="label">3 — Score Candidates</div>
      <div class="detail">
        IQL value score (long-horizon quality)<br/>
        + Inverse-dynamics self-consistency score
      </div>
    </div>
  </div>

  <div class="arrow-down"><span></span></div>

  <div class="row">
    <div class="box process wide">
      <div class="label">4 — Pick Best Action & Execute</div>
      <div class="detail">Highest combined score wins → send to vehicle → loop back with next real frame</div>
    </div>
  </div>

  <div class="arrow-down"><span></span></div>

  <div class="section-title green">Outputs / Evaluation</div>
  <div class="row">
    <div class="box output">
      <div class="label">Predicted Future Frames</div>
      <div class="detail">4-frame rollouts at 128×128</div>
    </div>
    <div class="box output">
      <div class="label">Selected Ego-Action</div>
      <div class="detail">Steering + throttle sent to vehicle</div>
    </div>
    <div class="box output">
      <div class="label">Evaluation Metrics</div>
      <div class="detail">FVD · Collision rate · Off-road rate · Progress-to-goal</div>
    </div>
  </div>

</div>

<!-- Legend -->
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#1e3a5f;border:1px solid #3b82f6"></div> Input</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#312e81;border:1px solid #818cf8"></div> Model Module</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#164e63;border:1px solid #22d3ee"></div> Process</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#4c1d2e;border:1px solid #f472b6"></div> Training Loss</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#14532d;border:1px solid #4ade80"></div> Output</div>
</div>

</body>
</html>


=========================

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>DriveWAM — Project Report</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Georgia', 'Times New Roman', serif;
    background: #fff;
    color: #1a1a1a;
    max-width: 820px;
    margin: 0 auto;
    padding: 48px 32px 64px;
    line-height: 1.7;
    font-size: 15px;
  }

  /* ---- Typography ---- */
  h1 {
    font-size: 1.7rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 4px;
    color: #111;
  }
  .doc-subtitle {
    text-align: center;
    font-size: 0.9rem;
    color: #666;
    margin-bottom: 36px;
    font-style: italic;
  }
  h2 {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1e293b;
    margin: 32px 0 10px;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 4px;
  }
  h3 {
    font-size: 1rem;
    font-weight: 700;
    color: #334155;
    margin: 20px 0 6px;
  }
  p { margin-bottom: 12px; }
  ul, ol { margin: 8px 0 14px 24px; }
  li { margin-bottom: 6px; }
  strong { color: #0f172a; }

  /* ---- Diagram section ---- */
  .diagram-section {
    background: #0f172a;
    border-radius: 14px;
    padding: 28px 24px 20px;
    margin: 24px 0;
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  }
  .diagram-section h3 {
    color: #f8fafc;
    font-size: 0.95rem;
    text-align: center;
    margin-bottom: 16px;
  }

  .box {
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
    font-size: 0.8rem;
    line-height: 1.4;
    box-shadow: 0 2px 10px rgba(0,0,0,.3);
  }
  .box .label { font-weight: 700; font-size: 0.88rem; margin-bottom: 3px; }
  .box .detail { color: #cbd5e1; font-size: 0.75rem; }

  .input   { background: #1e3a5f; border: 1.5px solid #3b82f6; }
  .module  { background: #312e81; border: 1.5px solid #818cf8; }
  .process { background: #164e63; border: 1.5px solid #22d3ee; }
  .output  { background: #14532d; border: 1.5px solid #4ade80; }
  .loss    { background: #4c1d2e; border: 1.5px solid #f472b6; }

  .row {
    display: flex;
    justify-content: center;
    gap: 14px;
    margin-bottom: 8px;
    flex-wrap: wrap;
  }
  .row .box { flex: 0 1 210px; }

  .arrow-down {
    display: flex;
    justify-content: center;
    margin: 3px 0;
  }
  .arrow-down span {
    display: block;
    width: 2px;
    height: 22px;
    background: #475569;
    position: relative;
  }
  .arrow-down span::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 50%;
    transform: translateX(-50%);
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 7px solid #475569;
  }
  .arrow-label {
    text-align: center;
    font-size: 0.65rem;
    color: #64748b;
    margin: -1px 0 2px;
  }
  .section-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin: 14px 0 6px 2px;
  }
  .section-label.blue   { color: #60a5fa; }
  .section-label.violet { color: #a78bfa; }
  .section-label.cyan   { color: #22d3ee; }
  .section-label.green  { color: #4ade80; }
  .section-label.pink   { color: #f472b6; }

  .phase-label {
    text-align: center;
    font-size: 0.78rem;
    font-weight: 700;
    color: #94a3b8;
    margin-bottom: 8px;
    letter-spacing: 1px;
  }
  .phase-divider {
    border: none;
    border-top: 1px dashed #334155;
    margin: 20px 0 14px;
  }
  .wide { flex: 0 1 440px !important; }

  .legend {
    display: flex;
    gap: 16px;
    justify-content: center;
    margin-top: 18px;
    flex-wrap: wrap;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.7rem;
    color: #94a3b8;
  }
  .legend-swatch {
    width: 12px; height: 12px;
    border-radius: 3px;
  }

  /* ---- Print-friendly ---- */
  @media print {
    body { padding: 24px; font-size: 13px; }
    .diagram-section { break-inside: avoid; }
  }
</style>
</head>
<body>

<h1>DriveWAM — Project Report</h1>
<p class="doc-subtitle">Diffusion Transformer World-Action Model for AV Scene Prediction and Closed-Loop Planning</p>

<!-- ============================================================ -->
<h2>1 &nbsp; Problem Statement</h2>
<p>
Self-driving cars need to <strong>see what is happening on the road</strong> and <strong>decide what to do next</strong> — steer left, brake, accelerate, etc. Today's leading autonomous-vehicle (AV) companies (Waymo, Wayve, Google DeepMind) each build internal systems that combine these two tasks into a single model called a <strong>World-Action Model (WAM)</strong>. A WAM looks at the current camera view, imagines what the road will look like a fraction of a second into the future under different possible driving actions, and then picks the best action.
</p>
<p>
The problem is that <strong>no open-source, camera-only WAM exists that can run on a normal consumer GPU</strong>. All existing implementations are proprietary and require massive compute clusters. This project — <strong>DriveWAM</strong> — sets out to build one from scratch: a lightweight, fully open version that anyone with a single GPU can train, evaluate, and learn from.
</p>

<!-- ============================================================ -->
<h2>2 &nbsp; How It Relates to Computer Vision</h2>
<p>
At its core, DriveWAM is a <strong>computer-vision system</strong>. Every piece of data it consumes and produces is visual:
</p>
<ul>
  <li><strong>Visual understanding:</strong> A Vision Transformer (ViT) encodes raw camera images into compact vector representations — the same family of models used in image classification, object detection, and segmentation.</li>
  <li><strong>Generative modeling:</strong> A Diffusion Transformer (DiT) generates future camera frames. Diffusion models are the engine behind modern image and video generation (e.g., Stable Diffusion, DALL-E). Here they are applied to <em>predict what a real driving scene will look like</em> rather than to create artistic images.</li>
  <li><strong>Learned perception:</strong> The inverse-dynamics head learns whether two frames are physically consistent with a given driving action — a form of self-supervised visual reasoning.</li>
</ul>
<p>
The project sits at the intersection of <strong>visual representation learning, generative video prediction, and vision-based decision-making</strong> — three core pillars of the CS231n curriculum.
</p>

<!-- ============================================================ -->
<h2>3 &nbsp; Data Source</h2>
<p>
The project uses the <strong>nuScenes mini split</strong>, a publicly available autonomous-driving dataset:
</p>
<ul>
  <li><strong>700 labeled driving scenes</strong> captured in real urban environments.</li>
  <li><strong>Front camera only</strong>, recorded at 12 frames per second.</li>
  <li><strong>~4 GB total size</strong> — small enough to download and process on a laptop.</li>
  <li><strong>No credentials or institutional access required</strong> — anyone can download it.</li>
  <li>Each scene includes <strong>time-stamped ego-vehicle annotations</strong>: steering angle and acceleration at every frame, which the model uses as its "action" signal.</li>
</ul>

<!-- ============================================================ -->
<h2>4 &nbsp; Inputs and Outputs</h2>

<h3>4.1 &nbsp; Training Inputs</h3>
<ol>
  <li><strong>Front-camera frame at time t</strong> — a single 128×128 RGB image.</li>
  <li><strong>Ego-action</strong> — the car's steering angle and acceleration at time t, encoded as a learned 16-dimensional embedding.</li>
  <li><strong>Ground-truth future frames (t+1 through t+4)</strong> — the real next four camera images, used to compute the diffusion training loss.</li>
</ol>

<h3>4.2 &nbsp; Training Outputs (Losses)</h3>
<ol>
  <li><strong>DDPM diffusion loss</strong> — measures how well the model reconstructs the true future frames.</li>
  <li><strong>Inverse-dynamics loss</strong> — measures whether the model can correctly recover the action from a pair of encoded frames (self-supervised consistency check).</li>
  <li><strong>IQL value loss</strong> — trained separately on logged human driving trajectories to learn which states are "good" (offline reinforcement learning).</li>
</ol>

<h3>4.3 &nbsp; Inference Inputs</h3>
<ol>
  <li><strong>One real observed frame (t)</strong> — the live dashcam image.</li>
</ol>

<h3>4.4 &nbsp; Inference Outputs</h3>
<ol>
  <li><strong>Predicted future frames</strong> — four imagined frames for each candidate action (visual prediction).</li>
  <li><strong>Selected ego-action</strong> — the steering and throttle command sent to the vehicle.</li>
  <li><strong>Evaluation metrics</strong> — FVD (video quality), collision rate, off-road rate, and progress-to-goal over 50 test scenes.</li>
</ol>

<!-- ============================================================ -->
<h2>5 &nbsp; Architecture Diagram</h2>

<!-- ====== DIAGRAM START ====== -->
<div class="diagram-section">

  <!-- TRAINING -->
  <div class="phase-label">TRAINING PHASE</div>

  <div class="section-label blue">Inputs</div>
  <div class="row">
    <div class="box input">
      <div class="label">Front-Camera Frame (t)</div>
      <div class="detail">128×128 RGB from nuScenes</div>
    </div>
    <div class="box input">
      <div class="label">Ego-Action</div>
      <div class="detail">Steering angle + acceleration → 16-d embedding</div>
    </div>
    <div class="box input">
      <div class="label">GT Frames (t+1 … t+4)</div>
      <div class="detail">Real next 4 frames for loss</div>
    </div>
  </div>

  <div class="arrow-down"><span></span></div>

  <div class="section-label violet">Model Modules</div>
  <div class="row">
    <div class="box module">
      <div class="label">① ViT-S Frame Encoder</div>
      <div class="detail">Image → patch tokens; action embedding concatenated</div>
    </div>
  </div>

  <div class="arrow-down"><span></span></div>
  <div class="arrow-label">encoded frame_t + action_embedding</div>

  <div class="row">
    <div class="box module">
      <div class="label">② 4-Layer DiT</div>
      <div class="detail">Diffusion Transformer denoises → predicted frames t+1…t+4</div>
    </div>
  </div>

  <div class="arrow-down"><span></span></div>
  <div class="arrow-label">predicted frames + encoded representations</div>

  <div class="row">
    <div class="box module">
      <div class="label">③ Inverse Dynamics Head</div>
      <div class="detail">2-layer MLP: (enc_frame_t, enc_frame_{t+4}) → predicted action</div>
    </div>
  </div>

  <div class="arrow-down"><span></span></div>

  <div class="section-label pink">Training Losses</div>
  <div class="row">
    <div class="box loss">
      <div class="label">DDPM Loss</div>
      <div class="detail">Reconstruct GT frames</div>
    </div>
    <div class="box loss">
      <div class="label">Inverse Dynamics Loss</div>
      <div class="detail">Predicted action ≈ true action</div>
    </div>
    <div class="box loss">
      <div class="label">IQL Value Loss</div>
      <div class="detail">Offline RL on logged trajectories</div>
    </div>
  </div>

  <!-- INFERENCE -->
  <hr class="phase-divider" />
  <div class="phase-label">INFERENCE &amp; CLOSED-LOOP PLANNING</div>

  <div class="section-label blue">Input</div>
  <div class="row">
    <div class="box input">
      <div class="label">Real Frame (t)</div>
      <div class="detail">Live dashcam image</div>
    </div>
  </div>

  <div class="arrow-down"><span></span></div>

  <div class="section-label cyan">Planning Process</div>
  <div class="row">
    <div class="box process wide">
      <div class="label">1 — Propose K=3 Actions</div>
      <div class="detail">3 candidate steering / throttle pairs</div>
    </div>
  </div>

  <div class="arrow-down"><span></span></div>

  <div class="row">
    <div class="box process wide">
      <div class="label">2 — Imagine Futures (DiT)</div>
      <div class="detail">Generate predicted future rollouts for each action</div>
    </div>
  </div>

  <div class="arrow-down"><span></span></div>

  <div class="row">
    <div class="box process wide">
      <div class="label">3 — Score Candidates</div>
      <div class="detail">IQL value score + inverse-dynamics consistency</div>
    </div>
  </div>

  <div class="arrow-down"><span></span></div>

  <div class="row">
    <div class="box process wide">
      <div class="label">4 — Execute Best Action</div>
      <div class="detail">Send to vehicle → observe next real frame → loop</div>
    </div>
  </div>

  <div class="arrow-down"><span></span></div>

  <div class="section-label green">Outputs</div>
  <div class="row">
    <div class="box output">
      <div class="label">Predicted Frames</div>
      <div class="detail">4 frames at 128×128</div>
    </div>
    <div class="box output">
      <div class="label">Selected Action</div>
      <div class="detail">Steering + throttle</div>
    </div>
    <div class="box output">
      <div class="label">Eval Metrics</div>
      <div class="detail">FVD · Collision · Off-road · Progress</div>
    </div>
  </div>

  <div class="legend">
    <div class="legend-item"><div class="legend-swatch" style="background:#1e3a5f;border:1px solid #3b82f6"></div> Input</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#312e81;border:1px solid #818cf8"></div> Module</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#164e63;border:1px solid #22d3ee"></div> Process</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#4c1d2e;border:1px solid #f472b6"></div> Loss</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#14532d;border:1px solid #4ade80"></div> Output</div>
  </div>
</div>
<!-- ====== DIAGRAM END ====== -->

<!-- ============================================================ -->
<h2>6 &nbsp; Summary</h2>
<p>
DriveWAM takes <strong>dashcam video + driving actions</strong> as input and learns to <strong>imagine future road scenes</strong> using a diffusion transformer. At test time it uses those imagined futures to <strong>plan the best driving action</strong> in a closed loop — propose, imagine, score, act, repeat. The entire system is trained on publicly available nuScenes data, runs on a single consumer GPU, and is evaluated on both video-prediction quality and real driving-safety metrics. It is, at its heart, a computer-vision project: every signal the model reads and writes is a camera image, processed through modern vision architectures (ViT, DiT) and generative modeling (DDPM diffusion).
</p>

</body>
</html>



