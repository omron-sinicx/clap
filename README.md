<div align="center">
<h1>👏 CLAP: Direct VLM-to-VLA Adaptation via Language-Action Grounding</h1>

<p align="center">
    Yuri Ishitoya<sup>1,2,†,‡</sup> &nbsp;
    Jeremy Siburian<sup>1,3,†,‡</sup> &nbsp;
    Masashi Hamaya<sup>1</sup> &nbsp;
    Haruto Suzuki<sup>1,†</sup>
</p>
<p align="center">
    Kuniaki Saito<sup>1</sup> &nbsp;
    Toshihiko Fukushima<sup>1</sup> &nbsp;
    Cristian C. Beltran-Hernandez<sup>1</sup> &nbsp;
    Mai Nishimura<sup>1,‡</sup>
</p>

<p align="center">
    <sup>1</sup>OMRON SINIC X Corporation &nbsp;
    <sup>2</sup>Ochanomizu University &nbsp;
    <sup>3</sup>The University of Tokyo
</p>

<p align="center">
    <sup>†</sup>Work done during an internship at OMRON SINIC X &nbsp;&nbsp;
    <sup>‡</sup>Equal contribution
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2607.08974"><img src="https://img.shields.io/badge/arXiv-2607.08974-orange" alt="arXiv"></a>
    <a href="https://omron-sinicx.github.io/clap/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
    <a href="https://huggingface.co/collections/omron-sinicx/clap"><img src="https://img.shields.io/badge/-HuggingFace-3B4252?style=flat&logo=huggingface&logoColor=" alt="HuggingFace"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg" alt="License: CC BY-NC 4.0"></a>
    <a href="https://pixi.sh"><img src="https://img.shields.io/badge/🪄%20Pixi-0.84.1-yellow" alt="Pixi 0.84.1"></a>
</p>

</div>

---

## 📄 Abstract

Vision-language-action models (VLAs) inherit semantic capabilities from pretrained VLMs, yet large-scale post-training on robot data and architectural modifications can reshape the backbone so extensively that it becomes difficult to isolate what the VLM contributes to control.
Directly converting pretrained VLMs into VLAs with minimal architectural change offers a more transparent path to understanding how VLM capabilities transfer across model scales.
The core obstacle is output-distribution mismatch: predicting actions as bare numeric token sequences moves generation away from the VLM's pretrained language distribution, degrading the capabilities we seek to preserve.
To address this, we propose CLAP (Causal Language-Action Prediction), which prepends each numeric action sequence with a natural-language action description, causally conditioning precise action-token prediction on a language-action plan without modifying the backbone architecture.
With only 4–5 hours of training, 2B CLAP achieves 90.8% on LIBERO (+14.9 pt over VLA-0) and improves robustness on LIBERO-PRO under language, object, and spatial perturbations.
Ablations attribute the gains to action-grounded language semantics, and on a real UR5e robot 2B CLAP reaches 60% success against 15% for matched VLA-0.
We release CLAP at 0.8B, 2B, and 4B as an open-weight, multi-scale compact VLA family from a single VLM lineage, enabling controlled analysis of VLM-to-VLA capability transfer.

## 🧰 Prerequisites

- Linux (x86_64) and NVIDIA GPUs with a driver that supports CUDA 12.6
- [pixi](https://pixi.sh) ≥ 0.84.1. Install it with:

  ```bash
  curl -fsSL https://pixi.sh/install.sh | sh
  ```

  or, if pixi is already installed, update it to the latest version with `pixi self-update`.

The training scripts have been tested on 8 × H200 (141 GB) and 4 × H100 (80 GB).

---

## 📦 Installation

```bash
git clone --recurse-submodules https://github.com/omron-sinicx/clap.git
cd clap
git submodule update --init   # only if you cloned without --recurse-submodules
pixi install      # training environment
pixi install -a   # all environments, including LIBERO evaluation
```

Run `pixi task list` to see the available tasks.

---

## 🏋️ Training

```bash
pixi run train                         # default config: clap_0.8b_4gpu
pixi run train clap_2b                 # configs/clap_2b.yaml
pixi run train clap_0.8b_4gpu 4,5,6,7  # override the GPU IDs
```

The arguments are positional: `[config] [gpu-ids]`.
Configs are in [`configs/`](configs/): those ending in `_4gpu` are for 4 GPUs, the others for 8 GPUs.
There is no 4-GPU config for 4B, which does not fit on 4 × H100 (80 GB) with a usable batch size.

### 📈 Data logging (trackio)

Training metrics are logged locally with [trackio](https://github.com/gradio-app/trackio); no account is needed.
Open the dashboard with:

```bash
pixi run trackio show --project clap
```

Set `trackio: false` in the config to disable logging.

To follow training from a browser or share the dashboard, trackio can also log to (or sync local logs to) a Hugging Face Space, e.g. `TRACKIO_SPACE_ID=<hf-user>/clap-trackio pixi run train clap_2b` after `pixi run hf auth login`.
See the trackio documentation, [Deploying and Embedding Dashboards](https://huggingface.co/docs/trackio/deploy_embed), for public and private Spaces and for syncing logs recorded locally.

Per-epoch LIBERO results and training times for the 0.8B and 2B configs are in [docs/reproduction.md](docs/reproduction.md).

---

## 🤖 Evaluation (LIBERO)

Pretrained models are available on Hugging Face ([collection](https://huggingface.co/collections/omron-sinicx/clap)):
[CLAP-Qwen3.5-0.8B](https://huggingface.co/omron-sinicx/CLAP-Qwen3.5-0.8B),
[CLAP-Qwen3.5-2B](https://huggingface.co/omron-sinicx/CLAP-Qwen3.5-2B),
[CLAP-Qwen3.5-4B](https://huggingface.co/omron-sinicx/CLAP-Qwen3.5-4B).

```bash
pixi run eval                                                  # CLAP-Qwen3.5-0.8B on LIBERO-Spatial, 4 GPUs
pixi run eval omron-sinicx/CLAP-Qwen3.5-2B libero_10 8         # LIBERO-Long on 8 GPUs
pixi run eval outputs/<run>/epoch-7                            # your own checkpoint
```

The arguments are positional: `[model] [suite] [gpus] [envs-per-gpu]`.
`model` is a Hugging Face repo id (default `omron-sinicx/CLAP-Qwen3.5-0.8B`) or a local checkpoint, and `suite` is one of `libero_spatial`, `libero_object`, `libero_goal` and `libero_10`.
The tasks of the suite are split across the GPUs, and the logs and results are written to `eval/logs/<model>_<suite>_<timestamp>/` (set `LOG_DIR` to change `eval/logs`).

**Faster evaluation on 8 GPUs.** Each suite has 10 tasks with 50 episodes each. Give `gpus` as a list of 10 GPU IDs (one task per process; GPUs 0 and 1 run two processes) and use 50 envs, so every task finishes in a single batch of episodes:

```bash
for suite in libero_spatial libero_object libero_goal libero_10; do
  pixi run eval omron-sinicx/CLAP-Qwen3.5-0.8B $suite 0,1,2,3,4,5,6,7,0,1 50
done
```

With CLAP-Qwen3.5-0.8B on 8 × H200 (141 GB) this takes about 10 minutes per suite.
Run the suites one after another: starting several suites at once launches hundreds of simulator processes together, which can saturate the node and stall the evaluation.

Summarize the results as a table, also while an evaluation is still running:

```bash
pixi run libero-table                      # results under eval/logs
pixi run libero-table path/to/logs         # another log directory
pixi run libero-table eval/logs --markdown # plain markdown table, e.g. to paste into a paper or an issue
```

In a terminal the table is colored: the best value of each column is in bold green, and runs that are still in progress are shown in yellow with `*`, followed by a progress bar per running evaluation (episodes done, elapsed time and an estimate of the time left).
When the output is redirected (e.g. to a file), a markdown table is printed.

---

> [!NOTE]
> This repository supports training on 4 × H100 or 8 × H200 GPUs.
> For a fair comparison with VLA-0, the models in the CLAP paper were trained and evaluated on top of the original [VLA-0 repository](https://github.com/NVlabs/vla0) (8 × H200, batch size 16 per GPU, 17,000 steps; 3 to 5.5 hours depending on the model size).
> 8 epochs in this repository correspond to those 17,000 steps in the VLA-0 repository: both are about 17,100 updates of 128 samples.
> The configs here train for 7 epochs, which gave results on par with the paper in less time (see [docs/reproduction.md](docs/reproduction.md)).

---

## 📝 Citation

If you find this codebase useful, please consider citing:

```bibtex
@inproceedings{ishitoya2026clap,
  title={CLAP: Direct VLM-to-VLA Adaptation via Language-Action Grounding},
  author={Yuri Ishitoya and Jeremy Siburian and Masashi Hamaya and Haruto Suzuki and Kuniaki Saito and Toshihiko Fukushima and Cristian C. Beltran-Hernandez and Mai Nishimura},
  booktitle={10th Annual Conference on Robot Learning},
  year={2026}
}
```

---

## ⚖️ License

This repository is released under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE) and may be used, shared and adapted for **non-commercial purposes only**.
CLAP builds on the following projects, and we thank their authors for making them available:

- [VLA-0](https://github.com/NVlabs/vla0) (NVIDIA, CC BY-NC 4.0): parts of the training and inference code are adapted from VLA-0, and its data pipeline RoboVerse is vendored under `libs/RoboVerse`.
- [LAP](https://github.com/lihzha/lap) (Apache-2.0): the rule-based action-to-language phrasing in `clap/models/clap_language.py` is adapted from LAP.
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (MIT): the simulation benchmark, included as a git submodule via the [RLinf/LIBERO](https://github.com/RLinf/LIBERO) fork.
- [Qwen3.5](https://huggingface.co/Qwen) (Apache-2.0): the base vision-language models.

See [NOTICE](NOTICE) for the full attributions and the list of changes.
