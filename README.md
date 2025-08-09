# KTQPP: Transferring Knowledge from Ranking Models for Query Performance Prediction
KTQPP is a deep learning framework for Query Performance Prediction (QPP) that uses a knowledge transfer approach to transfer insights from powerful ranking models (like BERT-based rankers) to lightweight QPP models. By aligning the internal representations of queries and documents between a source (ranking model) and a target (QPP model), KTQPP improves the prediction of query difficulty without requiring large labeled datasets. It outperforms traditional and neural QPP baselines on multiple benchmark datasets.
### Source Model
To begin, utilize the PreTTR ranking model and train it on the MS MARCO dataset. This allows the model to encode query-document relationships in a latent embedding space, which can subsequently be transferred as shared knowledge to a target QPP model such as KTQPP.

You can use the following pretrained checkpoint as a starting point:
https://huggingface.co/sebastian-hofstaetter/prettr-distilbert-split_at_3-margin_mse-T2-msmarco

Once the PreTTR model has been successfully trained on MS MARCO, store the resulting source model in the ```source``` directory for use in the knowledge transferring process.

### Input
The input format for KTQPP follows the same structure as BERTQPP, utilizing serialized files such as ```train_mrr.pkl``` and ```test.pkl```, which contain pre-encoded query-document pairs along with their corresponding metadata. For further details on the data preparation and format, please refer to the official BERTQPP repository:
https://github.com/Narabzad/BERTQPP

### Training

Before training the KTQPP model, ensure that the path to the source ranker model is correctly specified in the ```model.py``` script. Subsequently, execute train.py to optimize the model for ```MRR@10```, using BM25 retrieval on the MS MARCO training set. Alternatively, the model can be trained with a custom evaluation metric by preparing the corresponding training ```.pkl``` file. Users may also adjust hyperparameters such as ```epoch_num```, ```batch_size```, ```alpha```, and the initial pre-trained model within the script. In our experiments, we employed the ```bert-base-uncased``` model. Upon completion, the trained model will be saved in the ```models/``` directory.
### Evaluate
To evaluate a trained KTQPP model, specify the desired checkpoint within the ```test.py``` script and execute it. The predicted query performance scores will be saved in the ```results/``` directory, formatted as: 

```QID\tPredicted_QPP_Value```. 

To assess the effectiveness of the predictions, compute the correlation between the predicted QPP values and the actual retrieval performance of the corresponding queries.
