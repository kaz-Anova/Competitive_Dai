# Competitive_Dai
The code to generate a top 20 score in the [amazon classification challenge](https://www.kaggle.com/c/amazon-employee-access-challenge) using [DAI's](https://www.h2o.ai/driverless-ai/) predictions and feature engineering : This is in connection with [this video](https://www.youtube.com/watch?v=qtUNyJlAID0&amp;t=11s), regarding getting competitive with Driverless AI,  

Note I had to re-run this and results are a few decimal points different (from 0.9158 to 0.9157). I have also used an old version  (1.0.3) to be consistent with the video. Newer versions are far superior. 

3 additional corrections. 

1) the counts' features inlcude BOTH train and test summed (not just test)
2) Interractions for sparse logistic regression go up to 5-way, but most are 3-way. In any case it does not take more than 20 minutes to find these with a forward cv-aproach (in *prepare_data.py*)
3) In case you want to produce better features with the newer version of DAI, you should put some **text** (like a prefix) in front of the features' integer codes to help DAI unerstand they are catgeorical (and not numerical) . 

To run use:

```
#unzip munged.zip  to get train_munged.csv and test_munged.csv. These are the outputs of DAI in terms of feature engineering
#train.csv and test.csv the standard competition csvs from the amazon competition : https://www.kaggle.com/c/amazon-employee-access-challenge
#inside meta_folder/ there are the DAI predictions as dai_preds.train.csv and dai_preds.test.csv

#installations

pip install lightgbm
pip install sklearn
pip install pandas
pip install xgboost
pip install numpy
pip install scipy

#base models
python model_ama_lightgbm_v1.py
python model_ama_et.py
python model_ama_logit.py
python model_ama_rf.py
python model_ama_lightgbm_v2.py
python model_ama_lightgbmreg_v1.py
python model_ama_lightgbmreg_v2.py
python model_ama_xg_v1.py
python model_ama_xg_v2.py
python model_ama_lightgbm_v1_counts.py
python prepare_data.py  # finds interractions
model_ama_logit_v1_sparse.py # runs logistic regression with interractions


#Blending/Stacking
model_ama_meta_blender.py # runs blends everything 

```

For more ideas around this competition and stacking, you may have a look at [this](https://github.com/kaz-Anova/ensemble_amazon)

You may also look at [StackNet too](https://github.com/kaz-Anova/StackNet/blob/master/example/example_amazon/EXAMPLE.MD)



