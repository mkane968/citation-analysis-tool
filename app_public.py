from flask import Flask, render_template, request, jsonify
import re
import nltk
from nltk.tokenize import sent_tokenize
import os
import time

# Try to import the public version of the rhetorical move classifier
try:
    from rhetorical_move_classifier_public import RhetoricalMoveClassifier
    print("Using public version of rhetorical move classifier")
except ImportError:
    # Fallback to the original if available
    try:
        from rhetorical_move_classifier import RhetoricalMoveClassifier
        print("Using original version of rhetorical move classifier")
    except ImportError:
        print("Error: No rhetorical move classifier available")
        # Define a simple fallback classifier
        class RhetoricalMoveClassifier:
            def predict_rhetorical_move(self, sentence):
                return "Unknown", 0.0

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)
classifier = RhetoricalMoveClassifier()

# Get version timestamp for cache busting
VERSION = int(time.time())

@app.route('/')
def index():
    return render_template('fixed_chart.html', version=VERSION)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'error': 'No text provided'})
    
    # Process the text
    sentences, sentence_info, citation_count, rhetorical_moves = process_text(text)
    
    # Calculate percentages
    total_sentences = len(sentences)
    if total_sentences > 0:
        reporting_percentage = (rhetorical_moves['reporting'] / total_sentences) * 100
        transforming_percentage = (rhetorical_moves['transforming'] / total_sentences) * 100
        evaluating_percentage = (rhetorical_moves['evaluating'] / total_sentences) * 100
        citation_percentage = (citation_count / total_sentences) * 100
    else:
        reporting_percentage = transforming_percentage = evaluating_percentage = citation_percentage = 0
    
    # Get unique citations
    sorted_unique_citations = get_unique_citations(sentence_info)
    
    return jsonify({
        'sentence_analysis': sentence_info,  # Changed from 'sentences' to 'sentence_analysis' to match frontend expectations
        'citation_count': citation_count,
        'total_sentences': total_sentences,
        'citation_percentage': round(citation_percentage, 1),
        'reporting_percentage': round(reporting_percentage, 1),
        'transforming_percentage': round(transforming_percentage, 1),
        'evaluating_percentage': round(evaluating_percentage, 1),
        'unique_citations': sorted_unique_citations
    })

def get_unique_citations(sentence_info):
    """Extract and sort unique citations from sentence info"""
    unique_citations = set()
    
    for info in sentence_info:
        if info['has_citation'] and info['citations']:
            for citation in info['citations']:
                unique_citations.add(citation)
    
    # Sort citations alphabetically
    return sorted(list(unique_citations))

def process_text(text):
    """Process the text to identify citations and rhetorical moves"""
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Initialize counters
    citation_count = 0
    rhetorical_moves = {
        'reporting': 0,
        'transforming': 0,
        'evaluating': 0
    }
    
    # Process each sentence
    sentence_info = []
    
    for sentence in sentences:
        # Skip section headers (typically short and end with no period)
        if is_section_header(sentence):
            continue
            
        # Detect citations in the sentence
        has_citation, citations = detect_citations(sentence)
        
        if has_citation:
            citation_count += 1
        
        # Analyze rhetorical move
        rhetorical_move, confidence = analyze_rhetorical_moves(sentence)
        
        # Update rhetorical move counters
        if rhetorical_move.lower() == 'reporting':
            rhetorical_moves['reporting'] += 1
        elif rhetorical_move.lower() == 'transforming':
            rhetorical_moves['transforming'] += 1
        elif rhetorical_move.lower() == 'evaluating':
            rhetorical_moves['evaluating'] += 1
        
        # Add sentence info to the list
        sentence_info.append({
            'text': sentence,
            'has_citation': has_citation,
            'citations': citations,
            'rhetorical_move': rhetorical_move.lower(),
            'confidence': confidence
        })
    
    return sentences, sentence_info, citation_count, rhetorical_moves

def is_section_header(text):
    """Check if the text is likely a section header"""
    # Section headers are typically short, may be all caps, and don't end with a period
    if len(text) < 50 and not text.strip().endswith('.'):
        # Check if it's all caps or starts with a number (like "1. Introduction")
        if text.isupper() or re.match(r'^\d+\.?\s+\w+', text):
            return True
    return False

def detect_citations(sentence):
    """Detect citations in a sentence"""
    has_citation = False
    citations = []
    
    # Pattern for parenthetical citations like (Smith, 2020) or (Smith et al., 2020)
    parenthetical_pattern = r'\(([^)]*?\d{4}[^)]*?)\)'
    
    # Pattern for narrative citations like "According to Smith (2020)" or "Smith (2020) argues"
    narrative_pattern = r'([A-Z][a-z]+)(?:\s+et\s+al\.?)?\s+\(\d{4}\)'
    
    # Pattern for possessive forms like "Smith's (2020) study"
    possessive_pattern = r'([A-Z][a-z]+(?:\'s)?)(?:\s+et\s+al\.?)?\s+\(\d{4}\)'
    
    # Find all parenthetical citations
    parenthetical_matches = re.finditer(parenthetical_pattern, sentence)
    for match in parenthetical_matches:
        has_citation = True
        citation_text = match.group(1).strip()
        citations.append(citation_text)
    
    # Find all narrative citations
    narrative_matches = re.finditer(narrative_pattern, sentence)
    for match in narrative_matches:
        has_citation = True
        author = match.group(1).strip()
        # Extract the year from the context
        year_match = re.search(r'\((\d{4})\)', sentence[match.start():match.start()+30])
        if year_match:
            year = year_match.group(1)
            citations.append(f"{author}, {year}")
        else:
            citations.append(author)
    
    # Find all possessive citations
    possessive_matches = re.finditer(possessive_pattern, sentence)
    for match in possessive_matches:
        has_citation = True
        author = match.group(1).strip()
        # Extract the year from the context
        year_match = re.search(r'\((\d{4})\)', sentence[match.start():match.start()+30])
        if year_match:
            year = year_match.group(1)
            citations.append(f"{author}, {year}")
        else:
            citations.append(author)
    
    # Check for demonstrative pronouns referring to previous citations
    demonstrative_patterns = [
        r'\bthis (study|research|paper|article|work|analysis|review|survey|investigation)\b',
        r'\bthese (studies|researchers|authors|findings|results|conclusions)\b',
        r'\btheir (study|research|paper|article|work|analysis|review|survey|investigation)\b'
    ]
    
    for pattern in demonstrative_patterns:
        if re.search(pattern, sentence, re.IGNORECASE):
            has_citation = True
            # Don't add a specific citation text for demonstrative references
            break
    
    # Check for author references without years
    author_reference_pattern = r'([A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)?)\s+(?:argues?|claims?|states?|notes?|suggests?|proposes?|contends?|asserts?|demonstrates?|shows?|reveals?|indicates?|finds?|concludes?|observes?)'
    
    author_matches = re.finditer(author_reference_pattern, sentence)
    for match in author_matches:
        has_citation = True
        author = match.group(1).strip()
        citations.append(author)
    
    return has_citation, citations

def analyze_rhetorical_moves(sentence):
    """Analyze the rhetorical move of a sentence"""
    # Use the classifier to predict the rhetorical move
    rhetorical_move, confidence = classifier.predict_rhetorical_move(sentence)
    
    # Enhanced pattern matching for clear cases
    sentence_lower = sentence.lower()
    
    # Clear reporting patterns
    if re.search(r'\baccording to\b|\bstates? that\b|\breports? that\b|\bnotes? that\b|\bmentions? that\b|\bsays? that\b|\bpointed? out\b|\bindicates? that\b', sentence_lower):
        return "Reporting", 0.9
    
    # Clear evaluating patterns with strong evaluative language
    if re.search(r'\bargues? that\b|\bclaims? that\b|\bcontends? that\b|\bfails to\b|\blacks\b|\boverlooked\b|\bignores\b|\bmisses\b|\bneglects\b|\bproblematic\b|\bflawed\b|\blimited\b|\binsufficient\b|\binadequate\b|\bdeficient\b', sentence_lower):
        return "Evaluating", 0.9
    
    # Return the classifier's prediction
    return rhetorical_move, confidence

@app.route('/sample_text')
def sample_text():
    """Return sample academic text for demonstration"""
    sample_path = os.path.join('static', 'test_text_sample.txt')
    
    try:
        with open(sample_path, 'r', encoding='utf-8') as f:
            sample_text = f.read()
        return jsonify({'text': sample_text})
    except Exception as e:
        return jsonify({'error': str(e), 'text': 'Error loading sample text'})

if __name__ == '__main__':
    app.run(debug=True)
