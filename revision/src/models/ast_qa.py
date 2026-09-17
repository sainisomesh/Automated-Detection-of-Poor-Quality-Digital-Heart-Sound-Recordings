import torch
import torch.nn as nn
from transformers import ASTForAudioClassification, ASTConfig

class ASTHeartQA(nn.Module):
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593", freeze_base=True):
        super().__init__()
        self.config = ASTConfig.from_pretrained(model_name)
        # Load the full pre-trained model (with its classification head)
        self.original_model = ASTForAudioClassification.from_pretrained(model_name)
        
        # AudioSet has 527 classes. We want to keep this knowledge.
        # But we also want a binary head for Heart vs Noise.
        
        # We'll use the [CLS] token output from the transformer encoder
        # The ASTForAudioClassification puts a head on top of it.
        # We can simulate accessing the base model features.
        
        self.encoder = self.original_model.audio_spectrogram_transformer
        
        # Re-create the original head (it's a layernorm + linear)
        # ASTForAudioClassification.classifier is the head
        # It takes the hidden state of the first token (CLS)
        
        self.original_classifier = self.original_model.classifier
        
        # New Binary Head for Quality Assurance (Heart vs Non-Heart)
        self.qa_classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1) # Logits for binary classification
        )
        
        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False
            # Original classifier usually kept frozen if we trust AudioSet classes?
            # Or we can fine-tune it. Let's freeze encoder only.

    def forward(self, input_values):
        # Pass through encoder
        outputs = self.encoder(input_values)
        
        # Get CLS token state (sequence_output[:, 0, :])
        # sequence_output is outputs.last_hidden_state
        cls_token_state = outputs.last_hidden_state[:, 0, :]
        
        # Original AudioSet logits (527 classes)
        original_logits = self.original_classifier(cls_token_state)
        
        # QA logits (1 class: IsHeartSound)
        qa_logits = self.qa_classifier(cls_token_state)
        
        return original_logits, qa_logits

if __name__ == "__main__":
    # Test instantiation
    model = ASTHeartQA()
    print("Model initialized successfully.")
    print(model)
