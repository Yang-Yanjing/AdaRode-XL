import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef,
    accuracy_score, roc_auc_score, precision_recall_fscore_support
)

from transformers import (
    RobertaTokenizer, XLNetTokenizer, T5Tokenizer, BertTokenizer
)

from NetStructure.roberta_model import RoBERTaDet
from NetStructure.xlnet_model import XLnetDet
from NetStructure.t5_model import T5Det
from NetStructure.bert_model import BertDet



MODEL_PATH = '/root/autodl-fs/xlnet-base-cased'  # 预训练模型路径
CHECKPOINT_PATH = '/root/autodl-fs/AdaRode_XL/Model/AdaPIK/AdaRodeXLSQLsample/model_epoch23_val_loss0.0489.pth'  # 微调后的权重
# ===================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_LABELS = 3
# TEST_DATA_PATH = './Data/AdaRode/AugmentedData.csv'
# TEST_DATA_PATH = './Data/AdaPIK/test_set.csv'
TEST_DATA_PATH = './Data/MIV/test_set.csv'


BATCH_SIZE = 32
MAX_LENGTH = 128

# ==== 加载模型和Tokenizer ====
tokenizer = XLNetTokenizer.from_pretrained(MODEL_PATH)
model = XLnetDet(MODEL_PATH, NUM_LABELS).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

# ==== Tokenize 测试数据 ====
def tokenize_texts(texts, tokenizer, max_length=128):
    return tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt"
    ).to(DEVICE)

df = pd.read_csv(TEST_DATA_PATH)
inputs = tokenize_texts(df['Text'].tolist(), tokenizer)
labels = torch.tensor(df['Label'].tolist()).to(DEVICE)
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==== 测试模块 ====
class TestModule:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def test_with_progress(self, test_dataloader):
        self.model.eval()
        predictions, true_labels = [], []
        correct_predictions = 0

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Testing"):
                input_ids, attention_mask, labels = (t.to(self.device) for t in batch)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = correct_predictions.double() / len(test_dataloader.dataset)
        print(f'\nTest Accuracy: {accuracy:.4f}')
        self.preds = np.array(predictions)
        self.label = np.array(true_labels)
        self.report_metrics()

    def report_metrics(self):
        self.print_classification("Multi-Class", average='macro')
        self.print_binary("XSS", 2)
        self.print_binary("SQL", 1)

    def print_classification(self, title, average='macro'):
        print(f"\n{'='*20}{title}{'='*20}")
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.label, self.preds, average=average)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        y_true = self.label
        y_pred = self.preds
        y_true_binary = [1 if y in [1, 2] else 0 for y in y_true]
        y_pred_binary = [1 if y != 0 else 0 for y in y_pred]

        int_precision = precision_score(y_true_binary, y_pred_binary)
        int_recall = recall_score(y_true_binary, y_pred_binary)
        int_f1 = f1_score(y_true_binary, y_pred_binary)

        print("="*20 + "Interception Evaluation" + "="*20)
        print(f'Interception Precision: {int_precision:.4f}')
        print(f'Interception Recall:    {int_recall:.4f}')
        print(f'Interception F1-score:  {int_f1:.4f}')

    def print_binary(self, name, label_value):
        print(f"\n{'='*20}{name}{'='*20}")
        y_true = (self.label == label_value).astype(int)
        y_pred = (self.preds == label_value).astype(int)
        precision = precision_score(y_true, y_pred, pos_label=1)
        recall = recall_score(y_true, y_pred, pos_label=1)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = float('nan')
        mcc = matthews_corrcoef(y_true, y_pred)

        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"AUC:       {auc:.4f}")
        print(f"MCC:       {mcc:.4f}")

# ==== 启动测试 ====
if __name__ == "__main__":
    tester = TestModule(model, DEVICE)
    tester.test_with_progress(dataloader)
