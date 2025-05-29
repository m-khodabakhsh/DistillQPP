# DistillQPP: Distilling Knowledge from Ranking Models for Query Performance Prediction
DistillQPP is a deep learning framework for Query Performance Prediction (QPP) that uses knowledge distillation to transfer insights from powerful ranking models (like BERT-based rankers) to lightweight QPP models. By aligning the internal representations of queries and documents between a teacher (ranking model) and a student (QPP model), DistillQPP improves the prediction of query difficulty without requiring large labeled datasets. It outperforms traditional and neural QPP baselines on multiple benchmark datasets.
# Teacher Model
To begin, utilize the PreTTR ranking model and train it on the MS MARCO dataset. This allows the model to encode query-document relationships in a latent embedding space, which can subsequently be distilled as shared knowledge to a student QPP model such as DistilKPP.

You can use the following pretrained checkpoint as a starting point:
https://huggingface.co/sebastian-hofstaetter/prettr-distilbert-split_at_3-margin_mse-T2-msmarco

Once the PreTTR model has been successfully trained on MS MARCO, store the resulting teacher model in the ```teacher``` directory for use in the knowledge distillation process.

# Input
The input format for DistillQPP follows the same structure as BERTQPP, utilizing serialized files such as ```train_mrr.pkl``` and ```test.pkl```, which contain pre-encoded query-document pairs along with their corresponding metadata. For further details on the data preparation and format, please refer to the official BERTQPP repository:
https://github.com/Narabzad/BERTQPP

# Training

To train the DistillQPP model, execute the train.py script to optimize for ```MRR@10``` using BM25 retrieval on the MS MARCO training set. Alternatively, custom evaluation metrics can be supported by generating the corresponding training .pkl file. Users may also modify key hyperparameters—such as ```epoch_num```, ```batch_size```, and the initial ```pre-trained model```—within the script. In our experiments, we adopted the ```bert-base-uncased``` model. Upon completion of training, the resulting model checkpoint will be stored in the ```models/``` directory.
# Evaluate
