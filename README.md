# eTrust

### [Project](https://sites.google.com/view/thudm-etrust) | [PDF](http://keg.cs.tsinghua.edu.cn/jietang/publications/TKDE19-Cen-et-al-Trust-Relationship-Prediction.pdf)

Trust Relationship Prediction in Alibaba E-Commerce Platform

[Yukuo Cen](https://sites.google.com/view/yukuocen), [Jing Zhang](https://xiaojingzi.github.io/), Gaofei Wang, [Yujie Qian](http://people.csail.mit.edu/yujieq/), Chuizheng Meng, Zonghong Dai, [Hongxia Yang](https://sites.google.com/site/hystatistics/home), [Jie Tang](http://keg.cs.tsinghua.edu.cn/jietang/).

Accepted to TKDE 2019!

## Prerequisites

- g++
- make

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/THUDM/eTrust
cd eTrust
```

### Dataset

- Epinion  [Source](https://www.cse.msu.edu/~tangjili/trust.html)
- Ciao. [Source](https://www.cse.msu.edu/~tangjili/trust.html)
- Advogato. [Source](http://www.trustlet.org/datasets/advogato/)

### Training

First you should use command `make` to compile the cpp source file and obtain the executable file.

#### Training on the existing datasets

You can use `./src/main -data <datafile> -edge <edgefile>` to train eTrust-s model. 

For example, you can use `./src/main -data ./data/epinion.dat -edge ./data/epinion.edgelist` to train on Epinion dataset. You can similarly train on Ciao and Advogato datasets.

#### Training on your own datasets

If you want to train eTrust on your own dataset, you should prepare the following two files:
- xxx.dat: This file consists of two components: edge-based features and triad-based features.
    - component 1: edge-based features (starting with the label +k or ?k denoting labeled train/test instance of the model where k is the class number)
    - component 2: triangles formed by three edges
- xxx.edgelist: Each line represents an edge, which contains two numbers `<node1> <node2>`. Each line in the edgelist file corresponds to the line with the same line number in the data file.

Under construction. If you have ANY difficulties to get things working in the above steps, feel free to open an issue. You can expect a reply within 24 hours.


## Cite

Please cite our paper if you find this code useful for your research:

```
@article{cen2019trust,
  title={Trust Relationship Prediction in Alibaba E-Commerce Platform},
  author={Cen, Yukuo and Zhang, Jing and Wang, Gaofei and Qian, Yujie and Meng, Chuizheng and Dai, Zonghong and Yang, Hongxia and Tang, Jie},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2019},
  publisher={IEEE}
}
```
