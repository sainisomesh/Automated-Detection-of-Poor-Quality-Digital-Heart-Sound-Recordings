import torch
import torch.nn as nn
from transformers import ASTForAudioClassification, ASTConfig

class ASTHeartQA(nn.Module):
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593", freeze_base=True):
        super().__init__()
        self.config = ASTConfig.from_pretrained(model_name)
        # Full pretrained model, including its 527-class AudioSet head
        self.original_model = ASTForAudioClassification.from_pretrained(model_name)
        
        # Transformer encoder; the [CLS] token output feeds both heads
        self.encoder = self.original_model.audio_spectrogram_transformer
        
        # Original AudioSet head (layernorm + linear on the CLS state)
        self.original_classifier = self.original_model.classifier
        
        # Binary quality head (acceptable heart sound vs. not)
        self.qa_classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1) # Logits for binary classification
        )
        
        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False
            # Only the encoder is frozen; the AudioSet head is unused in the loss

    def forward(self, input_values):
        outputs = self.encoder(input_values)
        
        # CLS token state
        cls_token_state = outputs.last_hidden_state[:, 0, :]
        
        # Original AudioSet logits (527 classes)
        original_logits = self.original_classifier(cls_token_state)
        
        # QA logit (1 = acceptable heart sound)
        qa_logits = self.qa_classifier(cls_token_state)
        
        return original_logits, qa_logits

if __name__ == "__main__":
    # Quick instantiation check
    model = ASTHeartQA()
    print("Model initialized successfully.")
    print(model)
