# CaCao
This is the official repository for the paper "Visually-Prompted Language Model for Fine-Grained Scene Graph Generation in an Open World"
![framework](framework.png)
# Complete code for CaCao and boosted SGG
Here we provide sample code for CaCao boosting SGG dataset in standard setting and open-world setting.

## Running Script Tutorial
	1. python adaptive_cluster.py # obtain initialized clusters for CaCao
	2. python cross_modal_tuning.py # obtain cross-modal prompt tuning models for better predicate boosting
	3. python fine_grained_mapping.py # establish the mapping from open-world boosted data to target predicates for enhancement
	4. python fine_grained_predicate_boosting.py # enhance the existing SGG dataset with our CaCao model in <pre_trained_visually_prompted_model>
## Enhancement Retrain for SGG
	bash train_motif_expand.sh # expand SGG with motif models
	bash train_vctree_expand.sh # expand SGG with vctree models
	bash train_trans_expand.sh # expand SGG with transformer models
  
	# test for CaCao enhanced models
	bash test_expand.sh
Complete code and data for the scene graph generation with CaCao and Epic will be released with a final version of the paper.
# Quantitative Analysis
![image](https://user-images.githubusercontent.com/48062034/204216822-2010dd00-f0d4-4d5a-9a94-437589c4f8ea.png)
# Qualitative Analysis
![visualization](figures/visualization.png)
![visualization](figures/open-world.png)
## Predicate Boosting
![image](https://user-images.githubusercontent.com/48062034/204218380-3e2eedea-0adb-4acf-b3b6-c574c9e2dbfd.png)
## Predicate Prediction Distribution
![image](https://user-images.githubusercontent.com/48062034/204217723-3c053991-3df8-45c0-b99b-a9830cc2319e.png)
![image](https://user-images.githubusercontent.com/48062034/204218044-93bcd22e-96da-4fe7-8fb1-dacd7646d563.png)
