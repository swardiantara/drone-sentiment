import os
import argparse

import torch
from torch import nn
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import visualization as viz
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--model_name_or_path", default='bert-base-cased', type=str,
                        help="Path to pre-trained model or shortcut name")
    
    args = parser.parse_args()
    args.model_path = str(args.model_name_or_path).split('/')[-1]
    args.output_dir = os.path.join('outputs', args.model_path)

    if not os.path.exists(args.output_dir):
        raise FileNotFoundError("The model is not found!")
    
    return args


class ModelWrapper(nn.Module):
    def __init__(self, model: AutoModelForSequenceClassification):
        super().__init__()
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
        
    def forward(self, inputs):
        # Forward pass returning logits
        outputs = self.model(inputs)
        return outputs.logits
    
    def predict_proba(self, text):
        # Tokenize input text
        inputs = self.tokenizer(text, 
                              return_tensors="pt",
                              padding=True,
                              truncation=True,
                              max_length=512)
        
        # Get model predictions
        outputs = self(inputs['input_ids'])
        probs = torch.softmax(outputs, dim=1)
        return probs.detach().numpy()

def analyze_attribution(model_wrapper, text, target_class=1):
    """
    Perform attribution analysis using Integrated Gradients
    """
    # Tokenize input
    inputs = model_wrapper.tokenizer(text, 
                                   return_tensors="pt",
                                   padding=True,
                                   truncation=True,
                                   max_length=512)
    
    # Create IntegratedGradients instance
    ig = IntegratedGradients(model_wrapper)
    
    # Calculate attributions
    attributions, delta = ig.attribute(inputs['input_ids'],
                                     target=target_class,
                                     return_convergence_delta=True)
    
    # Convert attributions to word importance scores
    tokens = model_wrapper.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions.detach().numpy()
    
    # Create word-attribution pairs
    word_attributions = list(zip(tokens, attributions))
    
    return word_attributions, delta

def visualize_attributions(word_attributions):
    """
    Visualize word attributions
    """
    words, scores = zip(*word_attributions)
    
    # Normalize scores
    scores = np.array(scores)
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    
    print("\nWord Importance Scores:")
    print("-" * 50)
    for word, score in zip(words, scores):
        if word not in ['[CLS]', '[SEP]', '[PAD]']:
            print(f"{word:<20} {score:.4f}")

def main():
    args = init_args()
    # Assuming you have your SimpleTransformers model as 'model'
    model = AutoModelForSequenceClassification(args.output_dir)
    wrapped_model = ModelWrapper(model)
    
    # Example text for analysis
    texts = [
        "Cannot start Self-Timer. Exposure time is too long",
        "Cannot track subject: Subject too Small. Get Closer and retry",
        "Compass Interference. Temp Max Altitude: nnn",
        "No GPS signal. Unable to hover. Fly with caution",
        "Critical low battery voltage",
    ]
    
    for text in texts:
        # Perform attribution analysis
        word_attributions, delta = analyze_attribution(wrapped_model, text)
        
        # Visualize results
        visualize_attributions(word_attributions)
    

if __name__ == "__main__":
    main()