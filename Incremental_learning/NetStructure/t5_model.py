import torch.nn as nn
from transformers import T5ForConditionalGeneration

class T5Det(nn.Module):
    def __init__(self, model_name='/root/autodl-fs/t5-base', num_labels=3):
        super(T5Det, self).__init__()
        self.num_labels = num_labels
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.t5.config.d_model, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output.permute(0, 2, 1)).squeeze(-1)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits
