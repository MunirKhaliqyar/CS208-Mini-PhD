# Mini-PhD Project: Single-Turn vs Multi-Turn Jailbreak Safety

This repository contains the paper, datasets, analysis scripts, generated figures, and reproducibility materials for my mini-PhD project.

## Research topic

**Do single-turn safety benchmarks actually predict multi-turn jailbreak vulnerability?**

## Thesis in one sentence

Standard direct single-turn safety benchmarks do **not** predict multi-turn jailbreak vulnerability very well on their own, but a richer low-cost screen based on `retryability`, `decomposability`, and `benign-indistinguishability` can predict hidden conversational risk better than direct one-turn refusal tests alone.

## Repository contents

- `Munir_Khaliqyar_MiniPhD_Final_Paper_v2.docx`
  Final paper in Word format.
- `Mini-PhD-Multiturn-Safety-Paper.md`
  Editable markdown draft of the paper.
- `mini_defense_outline.pdf`
  Course requirements for the mini-PhD thesis and defense.
- `multistage_adni.pdf`
  Sample publication-style paper used as formatting guidance.
- `scripts/`
  Python scripts used to generate figures and build the final paper.
- `figures/`
  Generated figures used in the paper.
- `analysis_outputs/`
  CSV summaries and extracted analysis outputs.
- `llm_multiturn_attacks-main/`
  Main jailbreak dataset and example attack results used for the safety analysis.
- `ChatAlpaca-main/`
  Multi-turn dialogue dataset used to support the argument that chatbot interaction is naturally conversational.
- `conture-main/`
  Turn-level dialogue quality dataset used to support the argument that behavior changes across turns.

## Original dataset sources

The three datasets included in this repository come from these original GitHub sources:

- `llm_multiturn_attacks-main`
  [https://github.com/Micdejc/llm_multiturn_attacks](https://github.com/Micdejc/llm_multiturn_attacks)
- `ChatAlpaca-main`
  [https://github.com/icip-cas/ChatAlpaca](https://github.com/icip-cas/ChatAlpaca)
- `conture-main`
  [https://github.com/alexa/conture](https://github.com/alexa/conture)

## Main question answered by the project

The project tests whether a chatbot that appears safe in a single-turn benchmark will also remain safe when the same harmful goal is pursued across a multi-turn conversation.

The answer supported by the current analysis is:

- Direct single-turn benchmarks are weak predictors of later multi-turn jailbreak failure.
- Many harmful prompt families look safe in one turn but fail by turn three.
- A cheap stressed single-turn screen improves prediction, but does not replace full multi-turn testing.

## Evaluation framework

The paper organizes the low-cost evaluation around three ideas:

- `Retryability`
  Whether a harmful goal is already close to succeeding under small single-turn changes such as paraphrase, retry, or historical reframing.
- `Decomposability`
  Whether the harmful goal can be spread across multiple turns and still succeed.
- `Benign-indistinguishability`
  Whether the attack can remain normal-looking or low-suspicion while it escalates across turns.

## Main scripts

- `scripts/generate_attack_success_figures.py`
  Generates overall multi-turn jailbreak success figures.
- `scripts/generate_dialog_dynamics_figures.py`
  Generates supporting figures from ChatAlpaca and ConTurE.
- `scripts/generate_prediction_proxy_figures.py`
  Generates the main prediction-focused figures for the thesis.
- `scripts/build_final_paper_docx.py`
  Builds the final `.docx` paper from the local analysis outputs and figures.

## How to reproduce the figures

From the repository root, run:

```powershell
python .\scripts\generate_attack_success_figures.py
python .\scripts\generate_dialog_dynamics_figures.py
python .\scripts\generate_prediction_proxy_figures.py
```

Generated outputs:

- Figures are written to `figures/`
- CSV summaries are written to `analysis_outputs/`

## How to rebuild the paper

```powershell
python .\scripts\build_final_paper_docx.py
```

This writes the final paper as:

- `Munir_Khaliqyar_MiniPhD_Final_Paper_v2.docx`

If you also export a PDF version for GitHub or submission, place it in the repository root alongside the `.docx`.

## Python dependencies

Install dependencies with:

```powershell
python -m pip install pandas matplotlib numpy pypdf python-docx
```

## Key results currently highlighted in the paper

- `24.6%` of prompt-model pairs are hidden failures: safe in direct single-turn evaluation, unsafe by turn three.
- Direct single-turn warning has high precision but low recall for later multi-turn jailbreaks.
- A cheap stressed single-turn screen improves recall while keeping fairly high precision.

## Reproducibility note

This repository is intended to satisfy the course reproducibility requirement by including:

- the paper
- the datasets used
- the figure-generation scripts
- generated figures
- analysis outputs
- a rebuild script for the final `.docx`
