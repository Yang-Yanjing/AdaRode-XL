import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import XLNetTokenizer, XLNetModel
from urllib.parse import quote
import re
import random
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

device = torch.device("cuda:5") 

class RodeXL(nn.Module):
    def __init__(self, model_name='/nvme2n1/YangYJworks/ADV/xlnet-base-cased', num_labels=2):
        super(RodeXL, self).__init__()
        self.num_labels = num_labels
        self.xlnet = XLNetModel.from_pretrained(model_name)
        device = torch.device("cuda:5") 
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
    def __init__(self):
        device = torch.device("cuda:5")  # 将模型设置在第5个GPU上
        model = RodeXL(num_labels=3)
        model.to(device) 
        tokenizer = XLNetTokenizer.from_pretrained('/nvme2n1/YangYJworks/ADV/xlnet-base-cased')
        model.load_state_dict(torch.load(f"/nvme2n1/YangYJworks/ADV/AdaRodia/Model/SIK/XLnet/model_epoch17_val_loss0.0060.pth", map_location=torch.device("cpu")))
        model.to(torch.device("cuda:5"))
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
    