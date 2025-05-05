"""
Rhetorical Move Classifier - Public Version
This version uses only rule-based classification and does not require ML models.
"""

# Define the rhetorical move categories
RHETORICAL_MOVES = {
    0: "Reporting",     # Directly reporting what a source says
    1: "Transforming",  # Paraphrasing or synthesizing source material
    2: "Evaluating"     # Critiquing, analyzing, or evaluating sources
}

class RhetoricalMoveClassifier:
    def __init__(self, models_path=None):
        """
        Initialize the classifier with enhanced rule-based patterns.
        This public version does not require ML models.
        """
        # No models to load in the public version
        pass
    
    def predict_rhetorical_move(self, sentence):
        """Predict the rhetorical move category for a sentence using rule-based classification"""
        return self.rule_based_classification(sentence)
    
    def rule_based_classification(self, sentence):
        """Enhanced rule-based classification for rhetorical moves"""
        sentence = sentence.lower()
        
        # Reporting indicators - expanded with more patterns
        reporting_patterns = [
            "according to", "stated", "states", "reported", "reports", "said", "says",
            "noted", "notes", "mentioned", "mentions", "pointed out", "points out",
            "indicated", "indicates", "found", "finds", "showed", "shows",
            "observed", "observes", "described", "describes", "explained", "explains",
            "highlighted", "highlights", "emphasized", "emphasizes", "discussed", "discusses",
            "documented", "documents", "demonstrated", "demonstrates", "illustrated", "illustrates",
            "presented", "presents", "revealed", "reveals", "cited", "cites",
            "identified", "identifies", "confirmed", "confirms", "acknowledged", "acknowledges",
            "suggested", "suggests", "proposed", "proposes", "hypothesized", "hypothesizes",
            "concluded", "concludes", "summarized", "summarizes", "posited", "posits",
            "wrote", "writes", "articulated", "articulates", "expressed", "expresses",
            "author", "authors", "researcher", "researchers", "scholar", "scholars",
            "study", "studies", "research", "paper", "article", "publication",
            "et al", "et al.", "and colleagues"
        ]
        
        # Transforming indicators - expanded
        transforming_patterns = [
            "synthesize", "synthesizes", "synthesized", "combine", "combines", "combined",
            "integrate", "integrates", "integrated", "merge", "merges", "merged",
            "blend", "blends", "blended", "incorporate", "incorporates", "incorporated",
            "adapt", "adapts", "adapted", "modify", "modifies", "modified",
            "transform", "transforms", "transformed", "convert", "converts", "converted",
            "paraphrase", "paraphrases", "paraphrased", "restate", "restates", "restated",
            "build on", "builds on", "built on", "extend", "extends", "extended",
            "expand", "expands", "expanded", "elaborate", "elaborates", "elaborated",
            "develop", "develops", "developed", "derive", "derives", "derived",
            "draw from", "draws from", "drew from", "build upon", "builds upon", "built upon",
            "synthesizing", "combining", "integrating", "merging", "blending", "incorporating",
            "adapting", "modifying", "transforming", "converting", "paraphrasing", "restating",
            "this research", "these findings", "this study", "these studies", "this work",
            "this approach", "this method", "this framework", "this model", "this theory",
            "this concept", "these concepts", "this perspective", "these perspectives",
            "this view", "these views", "this understanding", "these understandings"
        ]
        
        # Evaluating indicators - expanded
        evaluating_patterns = [
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
            "refute", "refutes", "refuted", "contradict", "contradicts", "contradicted",
            "challenge", "challenges", "challenged", "question", "questions", "questioned",
            "doubt", "doubts", "doubted", "contest", "contests", "contested",
            "reject", "rejects", "rejected", "accept", "accepts", "accepted",
            "endorse", "endorses", "endorsed", "approve", "approves", "approved",
            "disapprove", "disapproves", "disapproved", "validate", "validates", "validated",
            "invalidate", "invalidates", "invalidated", "confirm", "confirms", "confirmed",
            "disconfirm", "disconfirms", "disconfirmed", "verify", "verifies", "verified",
            "disprove", "disproves", "disproved", "corroborate", "corroborates", "corroborated",
            "however", "nevertheless", "nonetheless", "although", "despite", "in spite of",
            "conversely", "in contrast", "on the contrary", "on the other hand",
            "fails to", "failed to", "fail to", "lacks", "lacked", "lack",
            "overlooks", "overlooked", "overlook", "ignores", "ignored", "ignore",
            "misses", "missed", "miss", "neglects", "neglected", "neglect",
            "overestimates", "overestimated", "overestimate", "underestimates", "underestimated", "underestimate",
            "exaggerates", "exaggerated", "exaggerate", "minimizes", "minimized", "minimize",
            "problematic", "flawed", "limited", "insufficient", "inadequate", "deficient",
            "weak", "strong", "compelling", "convincing", "unconvincing", "persuasive",
            "unpersuasive", "rigorous", "thorough", "comprehensive", "superficial", "simplistic",
            "nuanced", "sophisticated", "complex", "reductive", "innovative", "novel",
            "significant", "important", "crucial", "critical", "essential", "valuable",
            "worthwhile", "useful", "helpful", "unhelpful", "beneficial", "detrimental"
        ]
        
        # Check for evaluating indicators first (they're more specific)
        if any(pattern in sentence for pattern in evaluating_patterns):
            return "Evaluating", 0.8
        
        # Then check for reporting indicators
        if any(pattern in sentence for pattern in reporting_patterns):
            return "Reporting", 0.8
        
        # Then check for transforming indicators
        if any(pattern in sentence for pattern in transforming_patterns):
            return "Transforming", 0.8
        
        # Check for citation patterns
        if "(" in sentence and ")" in sentence:
            # If there's a citation but no clear rhetorical move, default to Reporting
            return "Reporting", 0.6
        
        # Default to Transforming as the most common category
        return "Transforming", 0.5
