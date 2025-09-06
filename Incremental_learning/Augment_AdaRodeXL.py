import yaml
import torch
import pandas as pd
import numpy as np
import random
import os
from transformers import XLNetTokenizer, XLNetModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import pickle

data_path = "/root/autodl-fs/AdaRode_XL/Data/AdaRode/AugmentedData.csv"
model_path = "/root/autodl-fs/Model/XLnet.pth"
class ModelTrainer:
    def __init__(self, config_path):
        self.load_config(config_path)
        self.setup_environment()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = XLNetTokenizer.from_pretrained(self.pretrained_name)
        self.model = RodeXL(self.pretrained_name, num_labels=self.num_labels).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.model.to(torch.device("cuda:0"))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.data_preparation()

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.dataset_path = os.path.join(os.getcwd(),"Data",config['Dataset'],"train_set.csv")
        self.pretrained_name = config['Model']['PretrainedName']
        self.num_labels = config['Model']['NumLabels']
        self.epochs = config['Train']['Epoch']
        self.max_length = config['Train']['Max_length']
        self.lr = config['Train']['lr']
        self.batch_size = config['Train'].get('BatchSize', 32)  # Default to 32 if not set
        self.dataset_name = config['Dataset']
        self.model_dir = os.path.join("./Model", self.dataset_name, "AdaRodeXLSQLsample")
        os.makedirs(self.model_dir, exist_ok=True)
        print("=== Loaded Parameters ===")
        print(f"Dataset Path: {self.dataset_path}")
        print(f"Pretrained Model Name: {self.pretrained_name}")
        print(f"Number of Labels: {self.num_labels}")
        print(f"Epochs: {self.epochs}")
        print(f"Max Length: {self.max_length}")
        print(f"Learning Rate: {self.lr}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Dataset Name: {self.dataset_name}")
        print(f"Model Directory: {self.model_dir}")
        print("==========================")
        
    def setup_environment(self):
        seed_value = 42
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    
    def data_preparation(self):

        train_path = "/root/autodl-fs/AdaRode_XL/Data/AdaRode/AugmentedData.csv"
        val_path = os.path.join(os.getcwd(), "Data", self.dataset_name, "val_set.csv")

        # 读取训练数据
        train_data = pd.read_csv(train_path)

        # 分离出 0 和 1 类
        label_01 = train_data[train_data["Label"].isin([0, 1])].reset_index(drop=True)
        label_rest = train_data[~train_data["Label"].isin([0, 1])].reset_index(drop=True)

        # 高斯采样 10% 的 0/1 样本
        num_to_sample = int(len(label_01) * 0.1)
        indices = np.arange(len(label_01))

        # 高斯采样参数：均值为中间，标准差为长度的 1/3（可调）
        mean = len(label_01) / 2
        std = len(label_01) / 3
        gauss_indices = np.random.normal(loc=mean, scale=std, size=num_to_sample).astype(int)

        # 保证索引合法且唯一
        gauss_indices = np.clip(gauss_indices, 0, len(label_01) - 1)
        gauss_indices = np.unique(gauss_indices)[:num_to_sample]

        sampled_label_01 = label_01.iloc[gauss_indices]
        train_data_sampled = pd.concat([sampled_label_01, label_rest], ignore_index=True).sample(frac=1, random_state=42)

        # 编码输入
        train_texts = train_data_sampled["Text"].tolist()
        train_labels = train_data_sampled["Label"].tolist()

        train_inputs = self.tokenize_texts(train_texts)

        train_dataset = TensorDataset(
            train_inputs["input_ids"],
            train_inputs["attention_mask"],
            torch.tensor(train_labels).to(self.device)
        )

        # 读取验证数据
        val_data = pd.read_csv(val_path)
        val_texts = val_data['Text'].tolist()
        val_labels = val_data['Label'].tolist()
        val_inputs = self.tokenize_texts(val_texts)
        val_dataset = TensorDataset(
            val_inputs['input_ids'],
            val_inputs['attention_mask'],
            torch.tensor(val_labels).to(self.device)
        )
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def tokenize_texts(self, texts):
        return self.tokenizer(texts, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
    
    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1} / {self.epochs}')
            self.model.train()
            total_loss = 0
            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}"):
                self.optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                loss, _ = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(self.train_dataloader)
            print(f'Epoch {epoch + 1}/{self.epochs}, Training Loss: {avg_train_loss:.4f}')
            val_loss = self.evaluate()
            # if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Validation loss improved, saving model...")
            save_path = os.path.join(self.model_dir, f'model_epoch{epoch + 1}_val_loss{val_loss:.4f}.pth')
            torch.save(self.model.state_dict(), save_path)
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                input_ids, attention_mask, labels = batch
                loss, _ = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_dataloader)
        print("Average Validation Loss:", avg_loss)
        return avg_loss


class RodeXL(nn.Module):
    def __init__(self, model_name='xlnet-base-cased', num_labels=3):
        super(RodeXL, self).__init__()
        self.num_labels = num_labels  # 确保这里保存了num_labels
        self.xlnet = XLNetModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.xlnet.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 在这里使用self.num_labels没有问题
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits


# 使用示例
config_path = './Config/Train.yaml'
trainer = ModelTrainer(config_path)
trainer.train()
