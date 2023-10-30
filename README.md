# NiPC-BPR
This is the official implementation for our paper **Recommendation of Mix-and-Match Clothing by Modeling Indirect Personal Compatibility**, accepted by **ICMR'23**.

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
- Note that we conduct our experiment under two different setting with two datasets, and you can modify the configurations in **yaml** files in **config** folder, where **_RB** refers to Given TOP and Recommend Bottom and **_RT** refers to Given Bottom and Recommend Top.
- The model code is at `Models/BPRs/NiPCBPR.py`.

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