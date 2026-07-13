<div align="center">
<h1>CLAP: Direct VLM-to-VLA Adaptation via Language-Action Grounding</h1>

<p align="center">
    Yuri Ishitoya<sup>1,*,†</sup> &nbsp;
    Jeremy Siburian<sup>2,*,†</sup> &nbsp;
    Masashi Hamaya<sup>3</sup> &nbsp;
</p>
<p align="center">
    Kuniaki Saito<sup>3</sup> &nbsp;
    Cristian C. Beltran-Hernandez<sup>3</sup> &nbsp;
    Mai Nishimura<sup>3,†</sup>
</p>

<p align="center">
    <sup>1</sup>Ochanomizu University &nbsp;
    <sup>2</sup>The University of Tokyo &nbsp;
    <sup>3</sup>OMRON SINIC X Corporation
</p>

<p align="center">
    <sup>†</sup>Equal contribution &nbsp;&nbsp;
    <sup>*</sup>Work done as an intern at OMRON SINIC X Corporation
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2607.08974"><img src="https://img.shields.io/badge/arXiv-2607.08974-orange" alt="arXiv"></a>
</p>

</div>

---

## 📄 Abstract

Vision-language-action models (VLAs) inherit semantic capabilities from pretrained VLMs, yet large-scale post-training on robot data and architectural modifications can reshape the backbone so extensively that it becomes difficult to isolate what the VLM contributes to control.
Directly converting pretrained VLMs into VLAs with minimal architectural change offers a more transparent path to understanding how VLM capabilities transfer across model scales.
The core obstacle is output-distribution mismatch: predicting actions as bare numeric token sequences moves generation away from the VLM's pretrained language distribution, degrading the capabilities we seek to preserve.
To address this, we propose CLAP (Causal Language-Action Prediction), which prepends each numeric action sequence with a natural-language action description, causally conditioning precise action-token prediction on a language-action plan without modifying the backbone architecture.
With single-epoch fine-tuning alone, 2B CLAP achieves 90.8% on LIBERO (+14.9 pt over VLA-0) and improves robustness on LIBERO-PRO under language, object, and spatial perturbations.
We will release CLAP at 0.8B, 2B, and 4B as an open-weight, multi-scale compact VLA family from a single VLM lineage, enabling controlled analysis of VLM-to-VLA capability transfer.

## 🚧 Code

**Code coming soon.**
