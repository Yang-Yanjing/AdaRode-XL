import os, time, random, yaml, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    RobertaTokenizer, XLNetTokenizer, T5Tokenizer, BertTokenizer
)

from NetStructure.roberta_model import RoBERTaDet
from NetStructure.xlnet_model import XLnetDet
from NetStructure.t5_model import T5Det
from NetStructure.bert_model import BertDet

Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModelTrainer:
    def __init__(self, config_path):
        self.load_config(config_path)
        self.setup_environment()
        self.device = Device
        self.load_tokenizer_and_model()
        self.data_preparation()
        self.start_time = time.time()
        self.end_time = time.time()

    def get_runing_time(self):
        return self.end_time - self.start_time

    def load_tokenizer_and_model(self):
        if self.model_name == "RoBERTa":
            self.tokenizer = RobertaTokenizer.from_pretrained(self.pretrained_path)
            self.model = RoBERTaDet(self.pretrained_path, self.num_labels).to(self.device)
        elif self.model_name == "XLnet":
            self.tokenizer = XLNetTokenizer.from_pretrained(self.pretrained_path)
            self.model = XLnetDet(self.pretrained_path, self.num_labels).to(self.device)
        elif self.model_name == "T5":
            self.tokenizer = T5Tokenizer.from_pretrained(self.pretrained_path)
            self.model = T5Det(self.pretrained_path, self.num_labels).to(self.device)
        elif self.model_name == "BERT":
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_path)
            self.model = BertDet(self.pretrained_path, self.num_labels).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.model_name = config['Model']['Model_name']
        self.pretrained_path = config['Model']['PretrainedName']
        self.num_labels = config['Model']['NumLabels']
        self.epochs = config['Train']['Epoch']
        self.max_length = config['Train']['Max_length']
        self.lr = config['Train']['lr']
        self.batch_size = config['Train'].get('BatchSize', 32)
        self.dataset_name = config['Dataset']
        self.model_dir = os.path.join("./Model", self.dataset_name, self.model_name)
        os.makedirs(self.model_dir, exist_ok=True)

    def setup_environment(self):
        seed_value = 42
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    def data_preparation(self):
        train_path = os.path.join(os.getcwd(), "Data", self.dataset_name, "train_set.csv")
        val_path = os.path.join(os.getcwd(), "Data", self.dataset_name, "val_set.csv")

        # 读取训练数据
        train_data = pd.read_csv(train_path)
        train_texts = train_data['Text'].tolist()
        train_labels = train_data['Label'].tolist()
        train_inputs = self.tokenize_texts(train_texts)
        train_dataset = TensorDataset(
            train_inputs['input_ids'],
            train_inputs['attention_mask'],
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

        # 创建 DataLoader
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
            val_loss = self.evaluate()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(self.model_dir, f'model_epoch{epoch + 1}_val_loss{val_loss:.4f}.pth')
                torch.save(self.model.state_dict(), save_path)
                self.end_time = time.time()

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                input_ids, attention_mask, labels = batch
                loss, _ = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += loss.item()
        return total_loss / len(self.val_dataloader)
