import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import XLNetTokenizer, get_linear_schedule_with_warmup
import torch.optim as optim
from NetStructure.xlnet_model import XLnetDet
import os
from sklearn.model_selection import train_test_split

CONFIG = {
    "pretrained_path": "/root/autodl-fs/xlnet-base-cased",
    "pretrained_weight_path": "/root/autodl-fs/Model/XLnet.pth",
    "num_labels": 3,
    "epochs": 10,
    "max_length": 128,
    "lr": 2e-5,
    "batch_size": 64,
    "warmup_steps": 100,
    "dataset_path": "/root/autodl-fs/AdaRode_XL/Data/AdaRode/AugmentedData.csv",
    "save_dir": "/root/autodl-fs/AdaRode_XL/Model/AugmentedXLNet"
}

Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class XLNetAugmentedTrainer:
    def __init__(self):
        self.device = Device
        self.set_seed()
        self.tokenizer = XLNetTokenizer.from_pretrained(CONFIG["pretrained_path"])
        self.model = XLnetDet(CONFIG["pretrained_path"], CONFIG["num_labels"])
        self.model.load_state_dict(torch.load(CONFIG["pretrained_weight_path"], map_location=self.device))
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=CONFIG["lr"])
        os.makedirs(CONFIG["save_dir"], exist_ok=True)
        self.load_data()

        total_steps = len(self.train_loader) * CONFIG["epochs"]
        print(f"Total training steps: {total_steps}")

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=CONFIG["warmup_steps"],
            num_training_steps=total_steps
        )

    def set_seed(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def tokenize_texts(self, texts):
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_length"],
            return_tensors="pt"
        ).to(self.device)

    from sklearn.model_selection import train_test_split

    def load_data(self):
        df = pd.read_csv(CONFIG["dataset_path"])
        if 'Text' not in df.columns or 'Label' not in df.columns:
            raise ValueError("CSV must contain 'Text' and 'Label' columns")

        # 全部保留 Label == 2
        df_2 = df[df['Label'] == 2]

        # 对 Label == 0 和 1 采样 50%
        df_0_1 = df[df['Label'].isin([0, 1])]
        df_0_1_sampled = df_0_1.sample(frac=0.5, random_state=42)

        # 合并数据 & 打乱顺序
        df_balanced = pd.concat([df_2, df_0_1_sampled], ignore_index=True).sample(frac=1, random_state=42)

        texts = df_balanced['Text'].tolist()
        labels = df_balanced['Label'].tolist()

        # 划分训练/验证集，保持标签分布一致
        texts_train, texts_val, labels_train, labels_val = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # 编码
        train_inputs = self.tokenize_texts(texts_train)
        val_inputs = self.tokenize_texts(texts_val)

        # 构建 TensorDataset
        train_dataset = TensorDataset(
            train_inputs['input_ids'],
            train_inputs['attention_mask'],
            torch.tensor(labels_train).to(self.device)
        )
        val_dataset = TensorDataset(
            val_inputs['input_ids'],
            val_inputs['attention_mask'],
            torch.tensor(labels_val).to(self.device)
        )

        self.train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

        # 打印统计信息
        from collections import Counter
        print("✅ Data Sampling Summary:")
        print(" - Label distribution (ALL):", dict(Counter(df_balanced['Label'])))
        print(" - Train size:", len(train_dataset))
        print(" - Val size:", len(val_dataset))


    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids, attention_mask, labels = batch
                loss, _ = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self):
        best_val_loss = float("inf")
        for epoch in range(CONFIG["epochs"]):
            print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
            self.model.train()
            total_loss = 0
            for batch in tqdm(self.train_loader, desc=f"Training"):
                input_ids, attention_mask, labels = batch
                self.optimizer.zero_grad()
                loss, _ = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()

            val_loss = self.evaluate()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(CONFIG["save_dir"], f"xlnet_aug_epoch{epoch+1}_valloss{val_loss:.4f}.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"✅ Saved improved model to: {save_path}")

if __name__ == "__main__":
    trainer = XLNetAugmentedTrainer()
    trainer.train()
