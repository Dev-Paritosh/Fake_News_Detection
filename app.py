from flask import Flask, render_template, request
import numpy as np
# import transformers
from transformers import AutoModel, BertTokenizerFast
import torch
import torch.nn as nn
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

app = Flask(__name__)


# Load BERT model and tokenizer via HuggingFace Transformers
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

class BERT_Arch(nn.Module):
    def __init__(self, bert):  
      super(BERT_Arch, self).__init__()
      self.bert = bert   
      self.dropout = nn.Dropout(0.1)            # dropout layer
      self.relu =  nn.ReLU()                    # relu activation function
      self.fc1 = nn.Linear(768,512)             # dense layer 1
      self.fc2 = nn.Linear(512,2)               # dense layer 2 (Output layer)
      self.softmax = nn.LogSoftmax(dim=1)       # softmax activation function
    def forward(self, sent_id, mask):           # define the forward pass  
      cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
                                                # pass the inputs to the model
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)                           # output layer
      x = self.softmax(x)                       # apply softmax activation
      return x

n_model = BERT_Arch(bert)
path = 'c2_new_model_weights.pt'
n_model.load_state_dict(torch.load(path))


@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/prediction', methods =['POST'])
def home():
    text = request.form['text']
    
    # 
    MAX_LENGHT = 15
    tokens_unseen = tokenizer.batch_encode_plus(text,max_length = MAX_LENGHT,pad_to_max_length=True,truncation=True)

    unseen_seq = torch.tensor(tokens_unseen['input_ids'])
    unseen_mask = torch.tensor(tokens_unseen['attention_mask'])

    with torch.no_grad():
        preds = n_model(unseen_seq, unseen_mask)
        preds = preds.detach().cpu().numpy()

    preds = np.argmax(preds, axis = 1)
    # 
    return render_template('after.html', pred =preds)
    

if __name__ == "__main__":
    app.run(debug=True)

    