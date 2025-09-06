import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import XLNetTokenizer, XLNetModel
from urllib.parse import quote
import re
import random
import os

device = torch.device("cuda:0") 

class RodeXL(nn.Module):
    def __init__(self, model_name='/root/autodl-fs/xlnet-base-cased', num_labels=2):
        super(RodeXL, self).__init__()
        self.num_labels = num_labels
        self.xlnet = XLNetModel.from_pretrained(model_name)
        device = torch.device("cuda:0") 
        self.xlnet.to(device)
        self.classifier = nn.Linear(self.xlnet.config.hidden_size, num_labels)

    def getmodeldevice(self):
        return self.xlnet.device
    def forward(self, input_ids, embed=None, attention_mask=None, labels=None):
        # 获取XLNet的embedding输出
        outputs = None
        if embed is not None:
            outputs = self.xlnet(inputs_embeds=embed, attention_mask=attention_mask)
        else:
            outputs = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])  # 取CLS token的输出进行分类

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class RodeXLAdapter:
    def __init__(self, modelpath, pretrain_model_path = '/root/autodl-fs/xlnet-base-cased'):
        device = torch.device("cuda:0")  
        model = RodeXL(num_labels=3)
        model.to(device) 
        tokenizer = XLNetTokenizer.from_pretrained(pretrain_model_path)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            model.load_state_dict(torch.load(modelpath, map_location=torch.device("cpu")))


        model.to(torch.device("cuda:0"))
        self.model = model
        self.tokenizer = tokenizer

    def get_pred(self, input):
        # 用于获取预测结果
        res = self.tokenizer(input, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        input_ids, attention_mask = res["input_ids"], res["attention_mask"]
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            return predictions

    def get_prob(self, input):
        res = self.tokenizer(input, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        input_ids, attention_mask = res["input_ids"], res["attention_mask"]
        input_ids.to(device)
        attention_mask.to(device)
        # 用于获取预测概率
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        prob = F.softmax(outputs, dim=1)
        numpy_list = prob.cpu().numpy().flatten().tolist()

        # Calculate the sum of the remaining numbers
        return numpy_list
    
# xss = "<script>alert(1)<script>"
# victim_model = RodeXLAdapter()
# a = victim_model.get_prob(xss)
# print(a)


