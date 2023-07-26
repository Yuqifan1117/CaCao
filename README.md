# CaCao
This is the official repository for the paper "Visually-Prompted Language Model for Fine-Grained Scene Graph Generation in an Open World" (Accepted by ICCV 2023)
![framework](figures/architecture.png)
# Complete code for CaCao and boosted SGG
Here we provide sample code for CaCao boosting SGG dataset in standard setting and open-world setting.
# Enhanced fine-grained predicates for VG
Download the enhanced dataset for VG training, you can use this [Google drive link](https://drive.google.com/drive/folders/1WOeumjptstD7nZQJgkJiqbQo9A_05gkh?usp=sharing).
## Running Script Tutorial
```bash
python adaptive_cluster.py # obtain initialized clusters for CaCao
python fine_grained_mapping.py # establish the mapping from open-world boosted data to target predicates for enhancement
```
```bash
python cross_modal_tuning.py # obtain cross-modal prompt tuning models for better predicate boosting
python fine_grained_predicate_boosting.py # enhance the existing SGG dataset with our CaCao model in <pre_trained_visually_prompted_model>
```
## Enhancement Retrain for SGG
	bash train_motif_expand.sh # expand SGG with motif models
	bash train_vctree_expand.sh # expand SGG with vctree models
	bash train_trans_expand.sh # expand SGG with transformer models
  
	# test for CaCao enhanced models
	bash test_expand.sh
## Open-world Predicate SGG Generalization
	# Backbone w/o Epic 
	bash train.sh TRANSGLOVE_novel
	# Epic (+CaCao)
	bash train.sh PROMPT_novel
Complete code and extra benchmark dataset for the scene graph generation with CaCao and Epic will be released with a final version of the paper.
# Quantitative Analysis
![image](https://github.com/Yuqifan1117/CaCao/assets/48062034/edd8b9bf-9d00-4f0f-894d-fce3b631fea5)
# Qualitative Analysis
![visualization](figures/visualization.png)
![visualization](figures/open-world.png)
## Predicate Boosting
![image](https://user-images.githubusercontent.com/48062034/204218380-3e2eedea-0adb-4acf-b3b6-c574c9e2dbfd.png)
## Predicate Prediction Distribution
![image](https://user-images.githubusercontent.com/48062034/204217723-3c053991-3df8-45c0-b99b-a9830cc2319e.png)
![image](https://user-images.githubusercontent.com/48062034/204218044-93bcd22e-96da-4fe7-8fb1-dacd7646d563.png)

## 📜 Citation
If you find this work useful for your research, please cite our paper and star our git repo:
```bibtex
@article{yu2023visually,
  title={Visually-Prompted Language Model for Fine-Grained Scene Graph Generation in an Open World},
  author={Yu, Qifan and Li, Juncheng and Wu, Yu and Tang, Siliang and Ji, Wei and Zhuang, Yueting},
  journal={arXiv preprint arXiv:2303.13233},
  year={2023}
}
```
