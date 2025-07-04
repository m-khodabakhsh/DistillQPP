import pickle 
from scipy.stats import kendalltau,pearsonr

from model import CrossEncoder
trained_model="/models/tuned_model-ce_bert-base-uncased_e1_b8"

with open('test.pkl', 'rb') as f:


    q_map_first_doc_test=pickle.load(f)

sentences = []
map_value_test=[]
queries=[]
for key in q_map_first_doc_test:
    sentences.append([q_map_first_doc_test[key]["qtext"],q_map_first_doc_test[key]["doc_text"]])
    queries.append(key)

model = CrossEncoder(trained_model, num_labels=1)
scores=model.predict(sentences)
actual=[]
predicted=[]

out=open('results/DistillQPP-score','w')

for i in range(len(sentences)):
    predicted.append(float(scores[i]))
    out.write(queries[i]+'\t'+str(predicted[i])+'\n')
out.close()
