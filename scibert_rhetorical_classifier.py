#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SciBERT Rhetorical Move Classifier

This module provides a SciBERT-based classifier for rhetorical moves that can be used
as a drop-in replacement for the existing TF-IDF classifier in the SourceMapper interface.
"""

import os
import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define the rhetorical move categories
RHETORICAL_MOVES = {
    0: "Reporting",     # Directly reporting what a source says
    1: "Transforming",  # Paraphrasing or synthesizing source material
    2: "Evaluating",    # Critiquing, analyzing, or evaluating sources
    3: "None"          # No citation or rhetorical move
}

class RhetoricalMoveClassifier:
    """
    SciBERT-based rhetorical move classifier that maintains compatibility
    with the existing SourceMapper interface.
    """
    
    def __init__(self, model_path="bert_comparison_results/scibert/final_model/"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize empty attributes to avoid any inheritance issues
        self.models = {}
        self.vectorizer = None
        self.ensemble = None
        
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned SciBERT model."""
        try:
            print(f"Loading SciBERT model from {self.model_path}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ SciBERT model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"✗ Error loading SciBERT model: {e}")
            print("Falling back to dummy predictions")
            self.model = None
            self.tokenizer = None
    
    def predict_rhetorical_move(self, sentence):
        """
        Predict the rhetorical move for a sentence.
        
        Args:
            sentence (str): The sentence to classify
            
        Returns:
            tuple: (predicted_move, confidence_score)
        """
        if self.model is None or self.tokenizer is None:
            # Fallback prediction if model failed to load
            return "Transforming", 0.5
        
        try:
            # Preprocess the sentence
            sentence = str(sentence).strip()
            if not sentence:
                return "None", 1.0
            
            # Tokenize
            inputs = self.tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probabilities = torch.softmax(logits, dim=-1)
                
                # Get predicted class and confidence
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = torch.max(probabilities).item()
            
            # Map to rhetorical move name
            predicted_move = RHETORICAL_MOVES.get(predicted_class, "Transforming")
            
            return predicted_move, confidence
            
        except Exception as e:
            print(f"SciBERT prediction error: {e}")
            print(f"Sentence length: {len(sentence)}")
            print(f"Device: {self.device}")
            import traceback
            traceback.print_exc()
            # Fallback prediction
            return "Transforming", 0.5
    
    def predict_rhetorical_move_no_none(self, sentence):
        """
        Predict rhetorical move excluding 'None' class - forces prediction among the 3 rhetorical moves.
        """
        try:
            # Tokenize the sentence
            inputs = self.tokenizer(
                sentence, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probabilities = torch.softmax(logits, dim=-1)
                
                # Exclude the "None" class (index 3) and renormalize
                rhetorical_probs = probabilities[0][:3]  # Only first 3 classes
                rhetorical_probs = rhetorical_probs / rhetorical_probs.sum()  # Renormalize
                
                # Get predicted class and confidence from the 3 rhetorical moves
                predicted_class = torch.argmax(rhetorical_probs).item()
                confidence = torch.max(rhetorical_probs).item()
            
            # Map to rhetorical move name (0=Reporting, 1=Transforming, 2=Evaluating)
            predicted_move = RHETORICAL_MOVES.get(predicted_class, "Transforming")
            
            return predicted_move, confidence
            
        except Exception as e:
            print(f"SciBERT no-none prediction error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback prediction
            return "Transforming", 0.5
    
    def predict_batch(self, sentences, batch_size=16):
        """
        Predict rhetorical moves for a batch of sentences.
        
        Args:
            sentences (list): List of sentences to classify
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of (predicted_move, confidence_score) tuples
        """
        if self.model is None or self.tokenizer is None:
            # Fallback predictions
            return [("Transforming", 0.5) for _ in sentences]
        
        results = []
        
        try:
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # Apply softmax to get probabilities
                    probabilities = torch.softmax(logits, dim=-1)
                    
                    # Get predicted classes and confidences
                    predicted_classes = torch.argmax(probabilities, dim=-1)
                    confidences = torch.max(probabilities, dim=-1)[0]
                
                # Convert to results
                for pred_class, confidence in zip(predicted_classes, confidences):
                    predicted_move = RHETORICAL_MOVES.get(pred_class.item(), "Transforming")
                    results.append((predicted_move, confidence.item()))
        
        except Exception as e:
            print(f"Error in batch prediction: {e}")
            # Fallback predictions
            results = [("Transforming", 0.5) for _ in sentences]
        
        return results
    
    def get_model_info(self):
        """Get information about the loaded model."""
        if self.model is None:
            return {
                'model_type': 'None (Failed to load)',
                'model_path': self.model_path,
                'device': str(self.device),
                'status': 'Error'
            }
        
        return {
            'model_type': 'SciBERT (allenai/scibert_scivocab_uncased)',
            'model_path': self.model_path,
            'device': str(self.device),
            'status': 'Loaded',
            'num_labels': len(RHETORICAL_MOVES)
        }

# For backward compatibility, also provide the old interface
def predict_rhetorical_move(sentence):
    """
    Standalone function for backward compatibility.
    """
    classifier = RhetoricalMoveClassifier()
    return classifier.predict_rhetorical_move(sentence)

if __name__ == "__main__":
    # Test the classifier
    classifier = RhetoricalMoveClassifier()
    
    test_sentences = [
        "According to Smith (2019), climate change is a major concern.",
        "The researchers found significant correlations in their data.",
        "However, this approach fails to consider important factors.",
        "This is a sentence without any citations."
    ]
    
    print("Testing SciBERT Rhetorical Move Classifier:")
    print("-" * 50)
    
    for sentence in test_sentences:
        move, confidence = classifier.predict_rhetorical_move(sentence)
        print(f"Sentence: {sentence[:60]}...")
        print(f"Predicted: {move} (confidence: {confidence:.3f})")
        print()
    
    print("Model Info:")
    print(classifier.get_model_info())
