# DistillQPP: Distilling Knowledge from Ranking Models for Query Performance Prediction
DistillQPP is a deep learning framework for Query Performance Prediction (QPP) that uses knowledge distillation to transfer insights from powerful ranking models (like BERT-based rankers) to lightweight QPP models. By aligning the internal representations of queries and documents between a teacher (ranking model) and a student (QPP model), DistillQPP improves the prediction of query difficulty without requiring large labeled datasets. It outperforms traditional and neural QPP baselines on multiple benchmark datasets.
# Teacher Model
To begin, utilize the PreTTR ranking model and train it on the MS MARCO dataset. This allows the model to encode query-document relationships in a latent embedding space, which can subsequently be distilled as shared knowledge to a student QPP model such as DistilKPP.

You can use the following pretrained checkpoint as a starting point:
https://huggingface.co/sebastian-hofstaetter/prettr-distilbert-split_at_3-margin_mse-T2-msmarco

Once the PreTTR model has been successfully trained on MS MARCO, store the resulting teacher model in the teacher directory for use in the knowledge distillation process.
