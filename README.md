# Dual Correlation Reduction Network

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://aaai.org/Conferences/AAAI-22/" alt="Conference">
        <img src="https://img.shields.io/badge/AAAI'22-brightgreen" /></a>
</p>

An official source code for paper *Deep Graph Clustering via Dual Correlation Reduction*, accepted by AAAI 2022.

### Overview

<p align = "justify"> 
    Deep graph clustering, which aims to reveal the underlying graph structure and divide the nodes into different groups, has attracted intensive attention in recent years. However, we observe that, in the process of node encoding, existing methods suffer from representation collapse which tends to map all data into a same representation. Consequently, the discriminative capability of node representations is limited, leading to unsatisfied clustering performance. To address this issue, we propose a novel self-supervised deep graph clustering method termed <b>D</b>ual <b>C</b>orrelation <b>R</b>eduction <b>N</b>etwork (<b>DCRN</b>) by reducing information correlation in a dual manner. Specifically, in our method, we first design a siamese network to encode samples. Then by forcing the cross-view sample correlation matrix and cross-view feature correlation matrix to approximate two identity matrices, respectively, we reduce the information correlation in dual level, thus improve the discriminative capability of the resulting features. Moreover, in order to alleviate representation collapse caused by over-smoothing in GCN, we introduce a propagation-regularization term to enable the network to gain long-distance information with shallow network structure. Extensive experimental results on six benchmark datasets demonstrate the effectiveness of the proposed DCRN against the existing state-of-the-art methods.
</p>



<div  align="center">    
    <img src="./assets/overall.jpg" width=60% />
</div>

