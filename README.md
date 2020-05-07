# Graph-Convolutional-Networks-With-Argument-Aware-Pooling-for-Event-Detection
implement model in paper Graph Convolutional Networks With Argument-Aware Pooling for Event Detection

# Preprocessing data:
- Transform format data from ACE format to json format from here: https://github.com/nlpcl-lab/ace2005-preprocessing.git
- build vocabularies : rebuild event vocab(BIO format in this project, include vocab[PAD] =-100), entity vocab( BIO format include pad label, id=0), word vocab( include pad token with id=0, unknow token with id =1) or just use vocabs from data folder
- use load_data_json and window_encode2 functions in utils.py to build data that will be fed into model
# Train model:
- hyperparameters of model are stored in Config class
- sample for training model have in file model.py( use EDModel2)
 
 # References: 
Graph Convolutional Networks With Argument-Aware Pooling for Event Detection, 
Thien Huu Nguyen, Ralph Grishman<br>
Jointly Multiple Events Extraction via Attention-based Graph Information Aggregation, 
Xiao Liu† and Zhunchen Luo‡ and Heyan Huang†∗
