from flask import Flask, render_template, request, jsonify, after_this_request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re

app = Flask(__name__)

def preprocess_text(text):
    # Temporarily replace "et al." to prevent sentence splitting
    text = text.replace("et al.", "et_al_PLACEHOLDER")
    
    # Split text into sentences
    sentences = nltk.sent_tokenize(text)
    
    # Restore "et al." in each sentence
    sentences = [s.replace("et_al_PLACEHOLDER", "et al.") for s in sentences]
    
    # Analyze each sentence for citations
    analyzed_sentences = []
    all_citations = []
    total_citation_count = 0
    
    for sentence in sentences:
        # Get citations from this sentence
        citations, citation_count = identify_citations(sentence)
        has_citation = citation_count > 0
        
        # Determine the citation style for the sentence
        citation_style = 'None'
        if has_citation:
            # Count citation styles
            style_counts = {}
            for citation in citations:
                style = citation.get('style', 'Unknown')
                style_counts[style] = style_counts.get(style, 0) + 1
            
            # Determine predominant style
            if len(style_counts) == 1:
                # Only one style present
                citation_style = list(style_counts.keys())[0]
            elif len(style_counts) > 1:
                # Multiple styles - find the most common
                max_count = 0
                for style, count in style_counts.items():
                    if count > max_count:
                        max_count = count
                        citation_style = style
                # If there's a tie, mark as Mixed
                if list(style_counts.values()).count(max_count) > 1:
                    citation_style = 'Mixed'
        
        # Extract just the citation text for display
        citation_texts = [citation['text'] for citation in citations]
        
        analyzed_sentences.append({
            'sentence': sentence,
            'has_citation': has_citation,
            'citations': citation_texts,
            'citation_count': citation_count,
            'citation_style': citation_style
        })
        
        all_citations.extend(citations)
        total_citation_count += citation_count
        
    return analyzed_sentences, all_citations, total_citation_count

def identify_citations(s):
    # Initialize the citations list with dictionaries containing text and style
    citations = []
    
    # Track years in APA citations to avoid duplication
    apa_years = set()
    
    # ===== APA CITATION FORMATS =====
    # 1. Single author: Smith (2020)
    for match in re.findall(r'([\w]+)\s+\((\d{4})\)', s):
        citations.append({
            'text': f"{match[0]} ({match[1]})",
            'style': 'APA'
        })
        apa_years.add(match[1])
    
    # 2. Two authors: Smith and Johnson (2020)
    for match in re.findall(r'([\w]+)\s+and\s+([\w]+)\s+\((\d{4})\)', s):
        citations.append({
            'text': f"{match[0]} and {match[1]} ({match[2]})",
            'style': 'APA'
        })
        apa_years.add(match[2])
    
    # 3. Three authors: Smith, Johnson, and Lee (2020)
    for match in re.findall(r'([\w]+),\s+([\w]+),\s+and\s+([\w]+)\s+\((\d{4})\)', s):
        citations.append({
            'text': f"{match[0]}, {match[1]}, and {match[2]} ({match[3]})",
            'style': 'APA'
        })
        apa_years.add(match[3])
    
    # 4. Four+ authors: Smith et al. (2020)
    for match in re.findall(r'([\w]+)\s+et\s+al\.\s+\((\d{4})\)', s):
        citations.append({
            'text': f"{match[0]} et al. ({match[1]})",
            'style': 'APA'
        })
        apa_years.add(match[1])
    
    # 5. Report with year: IPCC report (2022)
    for match in re.findall(r'([\w\s]+)\s+report\s+\((\d{4})\)', s, re.IGNORECASE):
        citations.append({
            'text': f"{match[0]} report ({match[1]})",
            'style': 'Report'
        })
        apa_years.add(match[1])
    
    # ===== MLA CITATION FORMATS =====
    # 1. One author with page number: (Johnson 42)
    for match in re.findall(r'\(([\w]+)\s+(\d+)\)', s):
        # Skip if this looks like an author with year that's already in APA citations
        if len(match[1]) == 4 and 1900 <= int(match[1]) <= 2100 and match[1] in apa_years:
            continue
        citations.append({
            'text': f"{match[0]} {match[1]}",
            'style': 'MLA'
        })
    
    # 2. Two authors with year: (Thompson and Lee 2018)
    for match in re.findall(r'\(([\w]+)\s+and\s+([\w]+)\s+(\d{4})\)', s):
        citations.append({
            'text': f"{match[0]} and {match[1]} {match[2]}",
            'style': 'MLA'
        })
        # Add to APA years to prevent duplication
        apa_years.add(match[2])
    
    # 3. Two authors without page numbers: (Smith and Johnson)
    for match in re.findall(r'\(([\w]+)\s+and\s+([\w]+)\)', s):
        citations.append({
            'text': f"{match[0]} and {match[1]}",
            'style': 'MLA'
        })
    
    # 4. Three authors with page numbers: (Chen, Roberts and Williams 78)
    for match in re.findall(r'\(([\w]+),\s+([\w]+)\s+and\s+([\w]+)\s+(\d+)\)', s):
        citations.append({
            'text': f"{match[0]}, {match[1]} and {match[2]} {match[3]}",
            'style': 'MLA'
        })
    
    # 5. Three authors without page numbers: (Smith, Johnson and Lee)
    for match in re.findall(r'\(([\w]+),\s+([\w]+)\s+and\s+([\w]+)\)', s):
        citations.append({
            'text': f"{match[0]}, {match[1]} and {match[2]}",
            'style': 'MLA'
        })
    
    # 6. Four+ author citations with page number: (Davis et al. 45)
    for match in re.findall(r'\(([\w]+)\s+et\s+al\.\s+(\d+)\)', s):
        citations.append({
            'text': f"{match[0]} et al. {match[1]}",
            'style': 'MLA'
        })
    
    # 7. Four+ author citations without page numbers: (Smith et al.)
    for match in re.findall(r'\(([\w]+)\s+et al\.\)', s):
        citations.append({
            'text': f"{match[0]} et al.",
            'style': 'MLA'
        })
    
    # 8. Citations with quoted material: ("Climate Report")
    for match in re.findall(r'\(\"(.*?)"\)', s):
        citations.append({
            'text': f'"{match}"',
            'style': 'MLA'
        })
    
    # 9. Single author without page number: (Smith)
    for match in re.findall(r'\(([\w]+)\)', s):
        # Skip if it's a year that's already in APA citations
        if match.isdigit() and len(match) == 4 and 1900 <= int(match) <= 2100 and match in apa_years:
            continue
        
        # Skip if this author is already part of another citation
        author_already_cited = False
        for citation in citations:
            if match == citation['text'].split()[0]:  # Check if it's the first word
                author_already_cited = True
                break
        
        if not author_already_cited:
            citations.append({
                'text': match,
                'style': 'MLA'
            })
    
    # 10. Narrative citations: As noted by Roberts
    for match in re.findall(r'(?:noted|mentioned|stated|cited|according to|as per)\s+by\s+([\w]+)', s, re.IGNORECASE):
        # Skip if this author is already part of another citation
        author_already_cited = False
        for citation in citations:
            if match == citation['text'].split()[0]:  # Check if it's the first word
                author_already_cited = True
                break
        
        if not author_already_cited:
            citations.append({
                'text': match,
                'style': 'Narrative'
            })
    
    # 11. Page number only citations: (42)
    for match in re.findall(r'\((\d+)\)', s):
        # Skip if it's a year that's already in APA citations
        if len(match) == 4 and 1900 <= int(match) <= 2100 and match in apa_years:
            continue
        
        # Check if this page number is already part of another citation
        page_already_cited = False
        for citation in citations:
            if f" {match}" in citation['text'] or match == citation['text']:
                page_already_cited = True
                break
        
        if not page_already_cited:
            citations.append({
                'text': f'p.{match}',
                'style': 'MLA'
            })
    
    # Better duplicate detection - check for partial matches
    unique_citations = []
    unique_citation_texts = []
    
    for i, citation in enumerate(citations):
        # Check if this citation is a subset of any other citation
        is_duplicate = False
        
        for j, other_citation in enumerate(citations):
            if i == j:  # Skip comparing with itself
                continue
                
            # Check if this citation is contained within another citation
            other_text = other_citation['text']
            if citation['text'] != other_text and citation['text'] in other_text:
                is_duplicate = True
                break
                
        if not is_duplicate and citation['text'] not in unique_citation_texts:
            unique_citations.append(citation)
            unique_citation_texts.append(citation['text'])
    
    # Sort citations by their position in the text for better presentation
    text_positions = []
    for citation in unique_citations:
        citation_text = citation['text']
        # Find position in the original text
        pos = s.find(citation_text.split('(')[0] if '(' in citation_text else citation_text)
        if pos == -1:  # If exact match not found, try a more flexible approach
            for part in citation_text.split():
                if len(part) > 3:  # Only consider substantial parts
                    pos = s.find(part)
                    if pos != -1:
                        break
        text_positions.append((pos if pos != -1 else float('inf'), citation))
    
    # Sort by position and extract just the citations
    sorted_unique_citations = [item[1] for item in sorted(text_positions, key=lambda x: x[0])]
    citation_count = len(sorted_unique_citations)
    
    return sorted_unique_citations, citation_count

def analyze_rhetorical_moves(sentence_info):
    # Simple rule-based approach to identify rhetorical moves
    sentence = sentence_info['sentence'].lower()
    has_citation = sentence_info['has_citation']
    
    # Check for different rhetorical moves
    if any(word in sentence for word in ['argue', 'argues', 'argued', 'claim', 'claims', 'claimed']):
        return 'Argument/Claim'
    elif any(word in sentence for word in ['find', 'finds', 'found', 'discover', 'discovers', 'discovered']):
        return 'Finding/Result'
    elif any(word in sentence for word in ['suggest', 'suggests', 'suggested', 'indicate', 'indicates', 'indicated']):
        return 'Suggestion/Indication'
    elif any(word in sentence for word in ['conclude', 'concludes', 'concluded', 'summary', 'summarize', 'summarized']):
        return 'Conclusion/Summary'
    elif any(word in sentence for word in ['according to', 'stated', 'states', 'reported', 'reports']):
        return 'Attribution'
    elif has_citation:
        return 'Citation Analysis'
    else:
        return 'Background/Context'

@app.route('/')
def home():
    @after_this_request
    def add_no_cache(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    # Serve the new template directly
    return render_template('index_new.html')

@app.route('/new')
def new_interface():
    @after_this_request
    def add_no_cache(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    return render_template('index_new.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    print('Received request:', request.json)
    text = request.json.get('text', '')
    
    if not text:
        return jsonify({
            'error': 'No text provided',
            'sentence_analysis': [],
            'citation_count': 0
        })
    
    # Preprocess and analyze the text
    analyzed_sentences, all_citations, total_citation_count = preprocess_text(text)
    
    # Add rhetorical move analysis
    for sentence_info in analyzed_sentences:
        sentence_info['rhetorical_move'] = analyze_rhetorical_moves(sentence_info)
        # Add confidence score (placeholder for now)
        sentence_info['confidence'] = 0.85  # Could be replaced with a real model confidence
    
    return jsonify({
        'sentence_analysis': analyzed_sentences,
        'citation_count': total_citation_count
    })

if __name__ == '__main__':
    nltk.download('punkt')  # Download required NLTK data
    app.run(debug=True, host='0.0.0.0', port=5001)
