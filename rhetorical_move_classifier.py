import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier

# Define the rhetorical move categories
RHETORICAL_MOVES = {
    0: "Reporting",     # Directly reporting what a source says
    1: "Transforming",  # Paraphrasing or synthesizing source material
    2: "Evaluating"     # Critiquing, analyzing, or evaluating sources
}

class RhetoricalMoveClassifier:
    def __init__(self, models_path="/Users/megankane/Documents/Source_Coding/"):
        self.models = {}
        self.vectorizer = None
        self.ensemble = None
        self.load_models(models_path)
        self.initialize_vectorizer()
        self.create_ensemble()
    
    def load_models(self, models_path):
        """Load the three machine learning models"""
        model_files = {
            "logistic_regression": f"{models_path}logistic_regression_model.pkl",
            "svm": f"{models_path}svm_model.pkl",
            "random_forest": f"{models_path}random_forest_model.pkl"
        }
        
        for name, path in model_files.items():
            try:
                with open(path, 'rb') as f:
                    self.models[name] = pickle.load(f)
                print(f"Loaded {name} model from {path}")
            except Exception as e:
                print(f"Error loading {name} model: {e}")
                self.models[name] = None
    
    def initialize_vectorizer(self):
        """Initialize the TF-IDF vectorizer with appropriate parameters"""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.85,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Since we don't have training data to fit the vectorizer,
        # we'll use a small set of common words in academic writing to initialize it
        common_academic_phrases = [
            "according to", "the author states", "argues that", "claims that",
            "suggests that", "proposes that", "contends that", "asserts that",
            "demonstrates that", "shows that", "reveals that", "indicates that",
            "finds that", "concludes that", "notes that", "observes that",
            "this suggests", "this indicates", "this implies", "this demonstrates",
            "this shows", "this reveals", "this proves", "this confirms",
            "in contrast to", "similar to", "unlike", "whereas", "while",
            "critically", "importantly", "significantly", "notably", "remarkably",
            "interestingly", "surprisingly", "as expected", "as anticipated",
            "evaluate", "assess", "analyze", "critique", "examine", "investigate",
            "explore", "consider", "review", "study", "research", "survey",
            "experiment", "observation", "finding", "result", "conclusion",
            "implication", "recommendation", "limitation", "strength", "weakness",
            "advantage", "disadvantage", "benefit", "drawback", "positive", "negative"
        ]
        self.vectorizer.fit(common_academic_phrases)
    
    def create_ensemble(self):
        """Create an ensemble of the loaded models"""
        # Filter out any models that failed to load
        valid_models = [(name, model) for name, model in self.models.items() if model is not None]
        
        if not valid_models:
            print("No valid models available for ensemble")
            return
        
        # Create a voting classifier from the loaded models
        self.ensemble = VotingClassifier(
            estimators=valid_models,
            voting='soft'  # Use probability estimates for voting
        )
    
    def predict_rhetorical_move(self, sentence):
        """Predict the rhetorical move category for a sentence"""
        # Check if we have any valid models
        if not any(model is not None for model in self.models.values()):
            # Fallback to rule-based classification if no models are available
            return self.rule_based_classification(sentence)
        
        # Vectorize the sentence
        features = self.vectorizer.transform([sentence])
        
        # If we have an ensemble, use it
        if self.ensemble is not None:
            try:
                # Try to use the ensemble for prediction
                probabilities = self.ensemble.predict_proba(features)[0]
                prediction = np.argmax(probabilities)
                confidence = probabilities[prediction]
                return RHETORICAL_MOVES[prediction], confidence
            except Exception as e:
                print(f"Error using ensemble: {e}")
        
        # If ensemble fails or isn't available, try individual models
        predictions = []
        confidences = []
        
        for name, model in self.models.items():
            if model is not None:
                try:
                    probas = model.predict_proba(features)[0]
                    pred = np.argmax(probas)
                    conf = probas[pred]
                    predictions.append(pred)
                    confidences.append(conf)
                except Exception as e:
                    print(f"Error using {name} model: {e}")
        
        if predictions:
            # Use the most confident prediction
            best_idx = np.argmax(confidences)
            best_prediction = predictions[best_idx]
            best_confidence = confidences[best_idx]
            return RHETORICAL_MOVES[best_prediction], best_confidence
        
        # Fallback to rule-based classification if all models fail
        return self.rule_based_classification(sentence)
    
    def rule_based_classification(self, sentence):
        """Rule-based classification as a fallback method"""
        sentence = sentence.lower()
        
        # Reporting indicators
        reporting_words = [
            "according to", "stated", "states", "reported", "reports", "said", "says",
            "noted", "notes", "mentioned", "mentions", "pointed out", "points out",
            "indicated", "indicates", "found", "finds", "showed", "shows"
        ]
        
        # Transforming indicators
        transforming_words = [
            "synthesize", "synthesizes", "synthesized", "combine", "combines", "combined",
            "integrate", "integrates", "integrated", "merge", "merges", "merged",
            "blend", "blends", "blended", "incorporate", "incorporates", "incorporated",
            "adapt", "adapts", "adapted", "modify", "modifies", "modified",
            "transform", "transforms", "transformed", "convert", "converts", "converted",
            "paraphrase", "paraphrases", "paraphrased", "restate", "restates", "restated"
        ]
        
        # Evaluating indicators
        evaluating_words = [
            "argue", "argues", "argued", "claim", "claims", "claimed",
            "suggest", "suggests", "suggested", "propose", "proposes", "proposed",
            "conclude", "concludes", "concluded", "recommend", "recommends", "recommended",
            "evaluate", "evaluates", "evaluated", "assess", "assesses", "assessed",
            "analyze", "analyzes", "analyzed", "critique", "critiques", "critiqued",
            "judge", "judges", "judged", "appraise", "appraises", "appraised",
            "criticize", "criticizes", "criticized", "praise", "praises", "praised",
            "support", "supports", "supported", "oppose", "opposes", "opposed",
            "agree", "agrees", "agreed", "disagree", "disagrees", "disagreed",
            "concur", "concurs", "concurred", "dispute", "disputes", "disputed",
            "refute", "refutes", "refuted", "contradict", "contradicts", "contradicted"
        ]
        
        # Check for indicators
        if any(word in sentence for word in reporting_words):
            return "Reporting", 0.7
        elif any(word in sentence for word in evaluating_words):
            return "Evaluating", 0.7
        elif any(word in sentence for word in transforming_words):
            return "Transforming", 0.7
        
        # If no clear indicators, check for citation patterns
        if "(" in sentence and ")" in sentence:
            return "Reporting", 0.6
        
        # Default to Transforming as the most common category
        return "Transforming", 0.5
