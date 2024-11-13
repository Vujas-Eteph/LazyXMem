# Lazy-XMem
Official Repository for LazyXMem

*[ACCV Paper]()*/ *[arXiv Paper](https://arxiv.org/pdf/2408.00169)* / *[Poster]()*

## TL;DR
- **Goals:** We aim to enhance long-term object tracking by prioritizing robustness over pure accuracy. Our approach leverages on-the-fly user corrections to improve performance while minimizing user workload. To evaluate this, we introduce **lazy video object segmentation** (**ziVOS**), where an object is segmented in a video with only one user interaction round (in contrast to interactive VOS), and where corrections are provided on-the-fly, *i.e,* while the method is segmenting the video sequence.
- **Motivations:** Lazy-XMem gauges prediction confidence in "real-time" (via Shannon Entropy) to determine when to use pseudo-corrections or request user input, with [SAM-HQ](https://github.com/SysCV/sam-hq) aiding the process. Pseudo-corrections reduce the need for user involvement by allowing self-correction.
- **Results:** Initial results are promising, showing performance and robustness gains through pseudo-corrections alone, and significant improvement with minimal user annotation (1.05% of the dataset).

## 📰 News:   
- Publishing the code
- Our paper got accepted at ACCV 2024! See you at Hanoi 🤗

## CODE

### Installation

#### Install Lazy-XMem
- Clone the Repository
    ```zsh
    git clone https://github.com/Vujas-Eteph/LazyXMem
    cd LazyXMem
    ```

- Create conda env. and package installation
    ```zsh
    conda env create -f environment.yml
    conda activate lazyXmem
    ```
- Install [Pytorch](https://pytorch.org/get-started/locally/) (follow the link)
- Download XMem weights
    ```zsh
    bash ./scripts/download_models.sh
    ```
- Installing SAM-HQ
    ```zsh
    git clone https://github.com/SysCV/sam-hq.git
    cd sam-hq; pip install -e .
    cd ..
    ```
- Download the [SAM-HQ weights](https://github.com/SysCV/sam-hq/issues/5) (follow the link)

- Download datasets (!todo: give informations)
    ```zsh

    ```

### How to run it?
Test Lazy-XMem: `eval_lazy_XMem_with_Lazy_agent.py` 
Test Lazy-XMem dummy: `eval_lazy_XMem_dummy.py` 
Test XMem: `eval_lazy_XMem_base.py` 

> [!TIPS]  
> Take a look at the `run.sh` file for a more detailed example.

### Other related projects:
The lazy agent simulator: https://github.com/Vujas-Eteph/iVOTS_robot
Tools for measuring the correlations and other cool stuff (⚠️ This project is currently a mess): *link*
