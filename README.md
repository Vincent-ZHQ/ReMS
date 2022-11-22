# ReMS
Code for Relevance-based Modality Specific Weighting for Multimodal Emotion Recognition

## 1. File system
\- models
<br> &ensp;  -- model.py
<br>
\- data
<br> &ensp;  -- dataloader.py
<br> &ensp;  -- train.pkl
<br> &ensp;  -- valid.pkl
<br> &ensp;  -- test.pkl
<br>
\- src
<br> &ensp;  -- utils
<br> &ensp;&ensp;  -- functions.py
<br> &ensp;&ensp;  -- metricsTop.py
<br> &ensp;  -- config.py
<br> &ensp;  -- train.py
<br>
\- results
<br>
\- main.py
<br>
\- requirements.txt

## 2. Environmet
- PyTorch version:  1.8.0
- CUDA version:  11.1
- cudnn version:  8005
- GPU:  Tesla V100-SXM2-16GB

## 3. How to use
 1. Downlioad pretrained Bert-base and Bert-large model from **https://huggingface.co/**
 2. Downlioad the data. 
 [Google Drive](https://drive.google.com/drive/folders/1VSY5BcAf8OgWV69DBC-zKmbynQTHyPR9?usp=sharing); 

 3. Install related libries. pip install requirements.txt

 4. Test. (2 examples) 
 - Bert-base: python main.py --name BaseTest --bert_type bert_base --rems_use --two_stage --test_only --seeds 3
 - Bert-large: python main.py --name LargeTest --bert_type bert_large --rems_use --two_stage --test_only --seeds 1

5. Train. (The predicting is not so stable, the following is an examples)
- python main.py --name RemsBase --num_workers 16 --bert_type bert_base --lr_other 1e-4 --post_other_dropout 0.0 --lr_text_bert 2e-5 --lr_text_other 1e-3 --post_text_dropout 0.0 --rems_use --two_stage --seeds 0 1 2 3 4

 


