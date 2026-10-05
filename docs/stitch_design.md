# Google Stitch design brief

The application UI was designed first in **Google Stitch** (https://stitch.withgoogle.com), then
implemented in React + Tailwind CSS. This file records the prompts used so the design can be
reproduced, and lists the screenshots that are kept as evidence in `docs/stitch/` and in the report.

## Prompt 1 – overall app shell and overview page (Web, desktop)

> Design a clean, professional web dashboard called **"GenAI Restoration Studio"** for a university
> generative-AI project (author tag "21i2525 · Assignment 1"). Light theme, slate-grey neutrals with an
> indigo (#4F46E5) accent, rounded 16px cards with subtle borders, Inter font, monospace for numbers.
> A sticky top bar has a square indigo "G" logo, the title, five navigation tabs – *Overview*,
> *Universal Restoration*, *Hard-Routed Restoration*, *Soft Mixture-of-Experts Restoration*,
> *Face-to-Sketch Generator* – and a pill-shaped "API ok" health badge with a green dot on the right.
> The Overview page has an indigo gradient hero banner explaining the four models, a 2×2 grid of
> clickable task cards (Task 1–4, title and one-line description), and a "System information" table
> listing each ONNX model file, its execution provider and a green "loaded" status chip.

## Prompt 2 – restoration workspace (used for Tasks 1–3)

> Design the **"Universal Restoration"** workspace for the same app. Two-column layout: a 340px left
> control column and a wide results area. Left column: a page title and short description; card
> "1 · Input image" with a segmented control (Sample / Upload) and a 4-column thumbnail gallery of pet
> photos or a dashed drag-and-drop zone; card "2 · Corruption" with a 2×2 grid of toggle buttons
> (None, Salt & pepper, Gaussian blur, Occlusion), pill buttons for severity (Low, Medium, High,
> Random, Custom) with a small monospace hint such as "k = 5, σ = 1.5", and a seed field; a full-width
> indigo "Run restoration" button. Results area: a row of four square image panels titled *Clean
> reference*, *Model input*, *Restored output* (highlighted with an indigo ring, with a Download link)
> and *|Error| map*; below them four small stat tiles (PSNR input, PSNR output with +dB delta, Source,
> Corruption), an "Inference time" card with millisecond tiles, and a collapsible JSON panel showing
> the corruption settings.

## Prompt 3 – routing panels (Tasks 2 and 3)

> Add to the restoration workspace a variant called **"Hard-Routed Restoration"**: under the
> corruption card, a segmented control "Routing mode: Predicted / Oracle". In the results area add a
> card "Classifier probabilities" with four horizontal bars (Clean grey, Salt & pepper amber,
> Gaussian blur sky-blue, Occlusion rose) showing percentages, the highest bar tagged "TOP", and a side
> card "Routing decision" with the predicted corruption, confidence, the selected expert, and a
> green "✓ Classifier correct" or red "✗ Misrouted" banner.
> Then a variant **"Soft Mixture-of-Experts Restoration"** with a card "Routing weights
> w = softmax(G(x̃)/τ)" containing four weight bars (Identity, S&P expert, Blur expert, Occlusion
> expert) and a stacked horizontal contribution bar, plus a "Gate behaviour" card showing the strongest
> contributor, whether routing is dominant or distributed (normalised entropy) and a ranking of the
> experts.

## Prompt 4 – face-to-sketch workspace

> Design the **"Face-to-Sketch Generator"** workspace: left column with card "1 · Facial photograph"
> (segmented control Sample / Upload / Webcam; the webcam view shows a live square preview with a
> dashed oval face guide and a "Capture photo" button), card "2 · Sketch style" with three large
> toggle buttons *Style 1*, *Style 2*, *Style 3*, and an indigo "Generate sketch" button. The results
> area shows the original photograph and the generated sketch side by side in two large square panels,
> a prominent "Download sketch" button, a style-condition stat tile and an inference-time card.

## Evidence to capture (save into `docs/stitch/`)

1. Screenshot of each generated Stitch screen (overview, universal, hard-routed, soft MoE, sketch).
2. Screenshot of the Stitch project page showing the prompts / iteration history.
3. Optional: the Stitch "export code" view, to show the design tokens that carried over (colours,
   radii, spacing) into `frontend/src/index.css`.

## Design → implementation mapping

| Stitch element | Implementation |
|---|---|
| Indigo accent, slate neutrals, 16px cards | `@theme` tokens and the `card` utility in `frontend/src/index.css` |
| Top navigation + health badge | `frontend/src/App.tsx` (`HealthBadge`) |
| Input card (Sample / Upload / Webcam) | `frontend/src/components/InputPicker.tsx` |
| Corruption + severity controls | `frontend/src/components/CorruptionControls.tsx` |
| Image panels, bar charts, stat tiles | `frontend/src/components/ui.tsx` |
| Restoration layout (Tasks 1–3) | `frontend/src/workspaces/RestorationShell.tsx`, `Workspaces.tsx` |
| Face-to-sketch layout | `frontend/src/workspaces/SketchWorkspace.tsx` |
