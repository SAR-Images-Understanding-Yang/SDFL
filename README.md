# Shortcut-Suppressed Domain-Generalized Feature Learning (SDFL)

## The core code will be available after paper get accepted!!!

Code for **Shortcut-Suppressed Domain-Generalized Feature Learning (SDFL)** for synthetic-to-measured SAR target recognition.

This package intentionally contains only three components used by the paper:

1. `sdfl.py` — SDFL training code.
2. `calibration/calibrate_parameters.py` — source-only calibration of `step_base` and `noise_base`.
3. `shortcut/shortcut_metrics.py` — shortcut-mechanism metrics used for the cross-domain reliability and matched-intervention analyses.


## 1. Environment

Install the packages in `requirements.txt` in a PyTorch environment with CUDA if GPU training is desired.

The released backbone is the same one-channel ResNet implementation used by the project. SDFL training initializes the backbone from ImageNet weights, while checkpoint-analysis scripts use `pretrained=False` because the loaded checkpoint overwrites the network weights.

## 2. Dataset layout

The three released benchmarks are:

- `SAMPLE`
- `S2M`
- `SimulatedSARShip`

Each dataset directory contains the source/target image-list files used by the original project:

```text
SampleDataset/
  Simulation.txt
  Real.txt
S2M/
  Simulation.txt
  Real.txt
SimulatedSARShip/
  Simulated.txt
  Real.txt
```

Each list line has the form

```text
/path/or/relative/path/to/image.png  class_id
```

## 3. SDFL training

Example for SAMPLE:

```bash
python sdfl.py /path/to/datasets/SampleDataset \
  -d SAMPLE -s S -t R -a resnet18 \
  -b 36 -j 0 --epochs 20 --seed 0 \
  --log /path/to/results/SAMPLE/sdfl/0
```

S2M uses the same calibrated SDFL parameters as SAMPLE. SimulatedSARShip automatically uses its calibrated ship-specific setting.

The released parameter values are:

| Dataset          | `step_base` | `noise_base` |
| ---------------- | ----------: | -----------: |
| SAMPLE           |         1.0 |          0.5 |
| S2M              |         1.0 |          0.5 |
| SimulatedSARShip |         0.4 |          0.3 |

The corresponding calibration CSV files are provided under `calibration/selected_parameters/`.


## 4. Source-only parameter calibration

The calibration code follows the source-only first-saturation protocol used in the paper. It does not use measured-domain recognition accuracy for selecting `step_base` or `noise_base`.

The calibration script expects independently trained source-only checkpoints at

```text
RESULTS_ROOT/<dataset>/source/<seed>/checkpoints/latest.pth
```

Run all three datasets with:

```bash
python calibration/calibrate_parameters.py \
  --datasets-root /path/to/datasets \
  --results-root /path/to/results \
  --datasets SAMPLE S2M SimulatedSARShip \
  --seeds 0-9 \
  --out calibration_output
```

## 5. Shortcut-mechanism metrics

The public shortcut diagnostic contains only the **Source vs. SDFL** mechanism analysis.

It implements the two mechanism-analysis stages used in the paper:

- **Cross-domain reliability:** classifier-response representations are extracted from the source model for global-amplitude, spatial, and scattering-intensity factors; source-trained multinomial logistic probes quantify how class predictivity changes from synthetic to measured SAR.
- **Matched intervention:** global-amplitude sensitivity is measured through fixed factor interventions, while spatial/scattering reliance is measured using matched Guided and Permuted interventions and prediction-flip based excess sensitivity.

Expected checkpoints are

```text
RESULTS_ROOT/<dataset>/source/<seed>/checkpoints/best.pth
RESULTS_ROOT/<dataset>/sdfl/<seed>/checkpoints/best.pth
```

Example:

```bash
python shortcut/shortcut_metrics.py \
  --datasets-root /path/to/datasets \
  --results-root /path/to/results \
  --datasets SAMPLE S2M SimulatedSARShip \
  --seeds 0-9 \
  --out shortcut_metrics_output
```

Use `--source-pattern` or `--sdfl-pattern` if your checkpoint directory names differ.
