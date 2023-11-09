# NiPC-BPR
This is the official implementation for our paper **Recommendation of Mix-and-Match Clothing by Modeling Indirect Personal Compatibility**, accepted by **ICMR'23**.<br/>
[paper](https://dl.acm.org/doi/abs/10.1145/3591106.3592224) 
***

> **Abstract:** Fashion recommendation considers both product similarity and compatibility, and has drawn increasing research interest. It is a challenging task because it often needs to use information from different sources, such as visual content or textual descriptions for the prediction of user preferences. In terms of complementary recommendation, existing approaches were dedicated to modeling either product compatibility or users’ personalization in a direct and decoupled manner, yet overlooked additional relations hidden within historical user-product interactions. In this paper, we propose a Normalized indirect Personal Compatibility modeling scheme based on Bayesian Personalized Ranking (NiPC-BPR) for mix-and-match clothing recommendations. We exploit direct and indirect personalization and compatibility relations from the user and product interactions, and effectively integrate various multi-modal data. Extensive experimental results on two benchmark datasets show that our method outperforms other methods by large margins.

<img src="https://d3i71xaburhd42.cloudfront.net/d0a6ad4f433422d4547775cbf5b1121362951f87/250px/3-Figure2-1.png">

***

## Requirements
The code is built on Pytorch library. Run the following code to satisfy the requeiremnts by pip:

`pip install -r requirements.txt`


## Datasets
- Download the two public datasets we use in the paper at:
  https://drive.google.com/file/d/1Dg7918zUGcL7tzs_OisNzc_FxYlQMG4E/view?usp=sharing

- Unzip the datasets and move them to **./dataset/**

- Unzip the well-trained models in saved and move them to **./saved/**

- You may also refer to the raw data here :)

  [IQON3000](https://drive.google.com/file/d/1sTfUoNPid9zG_MgV--lWZTBP1XZpmcK8/view)

  [Polyvore-U-519](https://stduestceducn-my.sharepoint.com/personal/zhilu_std_uestc_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzhilu%5Fstd%5Fuestc%5Fedu%5Fcn%2FDocuments%2Fpolyvore&ga=1)

## Train NiPCBPR
To train NiPCBPR with both visual and textual features, run the following script:

`python run_NiPCBPR.py`

## Evaluate
To evaluate the AUC of the well-trained model with default format (Given top and recommend bottom in Polyvore dataset):

`python test.py`

Or you can modify the corresponding path name.

## Tips
- Note that we conduct our experiment under two different setting with two datasets, and you can modify the configurations in **yaml** files in **./config/** folder, where **_RB** refers to Given TOP and Recommend Bottom and **_RT** refers to Given Bottom and Recommend Top.
- The model code is at `Models/BPRs/NiPCBPR.py`.

***

## Citation
If you find our work helpful, please kindly cite our research paper:
```
@inproceedings{liao2023nipcbpr,
  title={Recommendation of Mix-and-Match Clothing by Modeling Indirect Personal Compatibility},
  author={Shuiying Liao, Yujuan Ding and P. Y. Mok},
  booktitle={Proceedings of the 2023 ACM International Conference on Multimedia Retrieval (ICMR)},
  pages={560--564},
  year={2023}
}
```

 