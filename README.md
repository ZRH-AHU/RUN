# <p align=center> `RUN` <a href='https://arxiv.org/pdf/2501.18783'><img src='https://img.shields.io/badge/ArXiv-2501.18783-red'></a></p> 

This is the official PyTorch codes for the paper. 
>**RUN: Reversible Unfolding Network for Concealed Object Segmentation** <br> [Chunming He](https://chunminghe.github.io/), Rihan Zhang, Fengyang Xiao, [Chengyu Fang](https://cnyvfang.github.io/), Longxiang Tang, Yulun Zhang, Linghe Kong, Deng-Ping Fan, Kai Li, Sina Farsiu arXiv 2025<br>

**Abstract:** Existing concealed object segmentation (COS) methods frequently utilize reversible strategies to address uncertain regions. However, these approaches are typically restricted to the mask domain, leaving the potential of the RGB domain underexplored. To address this, we propose the Reversible Unfolding Network (RUN), which applies reversible strategies across both mask and RGB domains through a theoretically grounded framework, enabling accurate segmentation. RUN first formulates a novel COS model by incorporating an extra residual sparsity constraint to minimize segmentation uncertainties. The iterative optimization steps of the proposed model are then unfolded into a multistage network, with each step corresponding to a stage. Each stage of RUN consists of two reversible modules: the Segmentation-Oriented Foreground Separation (SOFS) module and the Reconstruction-Oriented Background Extraction (ROBE) module. SOFS applies the reversible strategy at the mask level and introduces Reversible State Space to capture non-local information. ROBE extends this to the RGB domain, employing a reconstruction network to address conflicting foreground and background regions identified as distortion-prone areas, which arise from their separate estimation by independent modules. As the stages progress, RUN gradually facilitates reversible modeling of foreground and background in both the mask and RGB domains, directing the network's attention to uncertain regions and mitigating false-positive and false-negative results. Extensive experiments demonstrate the superior performance of RUN and highlight the potential of unfolding-based frameworks for COS and other high-level vision tasks. We will release the code and models.   


<details>
<summary>🏃 The architecture of the proposed RUN</summary>
<center> 
    <img 
    src="featured.png">
</center>
</details>


## 🔥 News

- **2025-02-10:** We release this repository and the preprint of the full paper.


## 🔧 Todo

- [ ] Complete this repository



## 🔗 Contents

- [ ] Datasets
- [ ] Training
- [ ] Testing
- [ ] Results
- [ ] Citation
- [ ] Acknowledgements
