# 21i2525-A1 - Generative AI Assignment 1

Four related generative systems, from data pipeline to a containerised web application:

| Task | System | Workspace in the app | ONNX model(s) |
|---|---|---|---|
| 1 | Universal multi-corruption denoising autoencoder | **Universal Restoration** | `t1_universal_dae.onnx` |
| 2 | Corruption classifier + hard-routed specialist autoencoders | **Hard-Routed Restoration** | `t2_classifier.onnx`, `t2_specialist_{salt_pepper,blur,occlusion}.onnx` |
| 3 | Jointly trained soft mixture-of-experts | **Soft Mixture-of-Experts Restoration** | `t3_soft_moe.onnx` (full pipeline in one graph) |
| 4 | Style-conditioned face-to-sketch cGAN (pix2pix, FS2K) | **Face-to-Sketch Generator** | `t4_generator.onnx` |

Stack: PyTorch · Optuna (all 4 tasks) · MLflow · ONNX / ONNX Runtime · FastAPI · React + Tailwind CSS
(UI designed in Google Stitch, see [docs/stitch_design.md](docs/stitch_design.md)) · Docker Compose · uv.

## Results summary

| Task | Optuna | Test result | ONNX parity (max abs diff) |
|---|---|---|---|
| 1 Universal DAE | 30 trials | corrupted PSNR 19.3 → **24.6 dB**, SSIM 0.63 → 0.79 | 3.6e-6 |
| 2 Classifier + hard routing | 25 classifier trials; 15 + 12 specialist trials | classifier **99.82 % acc, macro-F1 0.997**; routing 24.3 dB, clean SSIM 0.997 | ≤ 1.2e-5 |
| 3 Soft MoE | 4 trials | 24.2 dB; gate top-1 matches the true label for 99.6 % of images; no inactive experts | 3.7e-6 |
| 4 Style-conditioned cGAN | 4 trials | FID 78.2, SSIM 0.45; three visibly distinct styles | 2.0e-6 |

Deadline note: the Task 3 and 4 Optuna studies (4 trials) and the final schedules (20-epoch specialists,
6-epoch MoE joint stage, 40-epoch GAN) are smaller than the budgets in the original design. See
`configs/*.yaml`, where they are marked `# reduced: deadline`.

---

## 1. Run the application (evaluators)

Requirements: **Docker Desktop** (with Compose) and **Git LFS** (bundled with Git for Windows; on other
systems install it from https://git-lfs.com). The trained ONNX models (≈114 MB) are stored with Git LFS. No Python, Node or IDE needed.

```bash
git lfs install
git clone https://github.com/Quttoshi/21i2525-A1.git
cd 21i2525-A1

git lfs pull            # fetches the ONNX models into ./models (no-op if the clone already got them)
docker compose up --build
```

Open **http://localhost:3000**. The FastAPI docs are at http://localhost:8000/docs.
Stop with `Ctrl+C` / `docker compose down`.

The backend checks every model at start-up. `GET /api/health` and the *System information* table on the
Overview page list each loaded ONNX model and its execution provider, or flag any file that is missing.

### API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | model load status, ONNX Runtime version / providers |
| GET | `/api/info` | model metadata, ONNX-vs-PyTorch parity reports |
| GET | `/api/samples` | bundled held-out sample images |
| POST | `/api/corrupt` | apply a corruption (type + severity, or custom parameters) |
| POST | `/api/restore/universal` | Task 1 |
| POST | `/api/restore/hard` | Task 2 (`routing=predicted|oracle`) |
| POST | `/api/restore/soft` | Task 3 |
| POST | `/api/sketch` | Task 4 (`style=1|2|3`) |

The restoration endpoints accept `file` (upload) **or** `sample`, plus `corruption`
(`none|salt_pepper|blur|occlusion`), `severity` (`low|medium|high|random|custom`), an optional
`seed`, and the custom parameters `p`, `kernel`, `sigma`, `n_rects` and `area`. Use `corruption=none` for an
image that is already corrupted.

---

## 2. Reproduce the research pipeline

Requirements: [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (developed on an RTX 3050 4 GB laptop
GPU with CUDA 12.8 wheels; CPU works but is slow).

```bash
uv sync                                         # creates .venv from pyproject.toml + uv.lock
uv run python scripts/download_data.py          # Oxford-IIIT Pet + FS2K, 128x128 caches, corruption manifests
uv run pytest                                   # corruption spec, manifest determinism, API tests
bash scripts/run_pipeline.sh                    # Optuna -> final training -> test eval -> ONNX export, all tasks
uv run python -m src.eval.plots                 # Optuna and training-curve figures for the report
uv run python scripts/make_samples.py           # sample gallery for the app
```

`run_pipeline.sh` skips any stage whose output already exists, so you can re-run it after an
interruption. You can also run stages one at a time:

```bash
uv run python -m src.train.t1_udae tune|train          uv run python -m src.eval.eval_t1
uv run python -m src.train.t2_classifier tune|train    uv run python -m src.train.t2_specialists tune|train
uv run python -m src.eval.eval_t2
uv run python -m src.train.t3_moe tune|train           uv run python -m src.eval.eval_t3
uv run python -m src.train.t4_cgan tune|train          uv run python -m src.eval.eval_t4
uv run python -m src.export.export_all                 # ONNX export + parity check -> models/parity_*.json
```

### Experiment tracking

All Optuna trials, final runs, hyperparameters, per-epoch losses and metrics, checkpoints, sample grids
and test evaluations are logged to MLflow (`mlruns/mlflow.db` + `mlruns/artifacts/`):

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000 --workers 1   # http://localhost:5000
```

The Optuna studies are SQLite files in `optuna_studies/`, one per study, so they can be reopened with
`optuna.load_study` or `optuna-dashboard`.

---

## 3. Data protocol (Tasks 1–3)

* Oxford-IIIT Pet: the official `trainval` set is split **80/20 train/val with seed 42**. The official `test`
  set is used only for the final evaluation, and the same split is used in Tasks 1, 2 and 3. Every image
  is converted to RGB and resized to 128×128.
* **Runtime corruption.** `RuntimeCorruptionDataset` samples a fresh condition (clean, salt-and-pepper, blur
  or occlusion, each with probability ¼) and fresh severity every time an image is loaded. No corrupted
  copies are written to disk. `BalancedBatchSampler` puts exactly B/4 samples of each condition into every
  batch.
* **Corruption definitions** ([src/data/corruptions.py](src/data/corruptions.py), NumPy-only and shared by
  training and the API):
  * salt-and-pepper: p ~ U(0.02, 0.15), black or white with equal probability;
  * Gaussian blur: k ∈ {3, 5, 7}, σ ~ U(0.5, 2.5), reflect padding (matches torchvision to < 1e-5);
  * occlusion: 1–3 non-overlapping black rectangles at random positions, together covering 10–35% of the
    image.
* **Deterministic manifests** ([data/manifests/](data/manifests/)). The validation manifest has one
  condition per image, balanced across conditions. The test manifest gives every test image clean plus 3
  fixed levels of each corruption: S&P 0.03/0.08/0.15; blur (3, 0.7)/(5, 1.5)/(7, 2.5); occlusion
  10%/20%/35% with 1/2/3 rectangles. Each entry stores the type, severity, seed, noise seed, blur
  parameters and rectangle coordinates.
* FS2K (Task 4): uses the official `anno_train.json` / `anno_test.json`, with a **15% style-stratified
  validation split (seed 42)** carved from the training portion. Photo and sketch receive identical paired
  augmentation (the same jitter crop and flip).

---

## 4. Repository layout

```
configs/                 per-task YAML: Optuna search spaces, schedules, defaults
src/data/                corruptions.py, pets.py (runtime corruption, balanced sampler), manifests.py, fs2k.py
src/models/              dae.py, classifier.py, moe.py, cgan.py
src/losses/              L1 + SSIM loss and image metrics
src/train/               t1_udae.py, t2_classifier.py, t2_specialists.py, t3_moe.py, t4_cgan.py (tune + train)
src/tune/                YAML -> Optuna search-space helper
src/eval/                eval_t1..t4.py, restoration.py, plots.py
src/export/              ONNX export + PyTorch/ONNX Runtime parity verification
backend/                 FastAPI + ONNX Runtime service, Dockerfile, sample images
frontend/                React + Vite + Tailwind CSS app, Dockerfile (nginx)
scripts/                 download_data.py, run_pipeline.sh, make_samples.py, make_dummy_models.py
tests/                   unit + API tests
data/manifests/          deterministic validation / test corruption manifests
optuna_studies/          Optuna SQLite studies
report/                  IEEE LaTeX report and figures
docs/                    Google Stitch design brief and evidence
docker-compose.yml       one-command deployment
```

Datasets, PyTorch checkpoints and MLflow artefacts are not committed (they are regenerated by the
pipeline). The ONNX models in `models/` (≈114 MB, mostly the 90 MB sketch generator) are tracked
with **Git LFS** (`.gitattributes`).

## 5. Acknowledgements and licences

* Oxford-IIIT Pet Dataset — Parkhi et al., CVPR 2012 (CC BY-SA 4.0).
* FS2K — Fan et al., "Facial-Sketch Synthesis: A New Challenge", MIR 2022 (research use).
* pix2pix — Isola et al., CVPR 2017. The U-Net/PatchGAN design follows the paper; the code is our own.
* SSIM via `pytorch-msssim`. FID uses torchvision's ImageNet Inception-v3 weights.
