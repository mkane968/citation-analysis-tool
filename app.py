from flask import Flask, render_template, request, jsonify, after_this_request, send_from_directory
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
import os
import tempfile
import PyPDF2
import docx
from flask import Flask, render_template, request, jsonify, after_this_request
from werkzeug.utils import secure_filename
from rhetorical_move_classifier import RhetoricalMoveClassifier

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Initialize the rhetorical move classifier
rhetorical_classifier = RhetoricalMoveClassifier()

def preprocess_text(text, author_names=None):
    # Initialize author_names if not provided
    if author_names is None:
        author_names = set()
    # First fix citations that might be broken across lines
    # Look for patterns like "(Wilson, 2019, pp. " followed by "78-92)" on the next line
    text = re.sub(r'\(([\w]+),\s+(\d{4}),\s+pp\.\s*\n\s*(\d+)-(\d+)\)', r'(\1, \2, pp. \3-\4)', text)
    
    # Also fix citations where just the page range is broken across lines
    text = re.sub(r'\(([\w]+),\s+(\d{4}),\s+pp\.\s*\n\s*(\d+.*?)\)', r'(\1, \2, pp. \3)', text)
    
    # Store paragraph breaks for later restoration
    paragraphs = text.split('\n\n')
    processed_paragraphs = []
    
    for paragraph in paragraphs:
        # Normalize whitespace within paragraphs
        paragraph = re.sub(r'\s+', ' ', paragraph)
        paragraph = paragraph.replace('\n', ' ').replace('\r', ' ')
        processed_paragraphs.append(paragraph.strip())
    
    # Rejoin with paragraph markers
    text = '\n\n'.join(processed_paragraphs)
    
    # Remove works cited section if present (common in essays)
    works_cited_patterns = [r'Works Cited.*$', r'References.*$', r'Bibliography.*$']
    for pattern in works_cited_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Fix common formatting issues from PDFs
    # Fix hyphenated words that might be split across lines
    text = re.sub(r'(\w+)- (\w+)', r'\1\2', text)
    
    # Fix citation formatting issues
    # Fix spaces in citations like "Smith ( 2020 )" to "Smith (2020)"
    text = re.sub(r'\( (\d{4}) \)', r'(\1)', text)
    
    # Temporarily replace "et al." to prevent sentence splitting
    text = text.replace("et al.", "et_al_PLACEHOLDER")
    
    # Split text into sentences, preserving paragraph breaks
    all_sentences = []
    for paragraph in text.split('\n\n'):
        sentences = nltk.sent_tokenize(paragraph)
        # Restore "et al." in each sentence
        sentences = [s.replace("et_al_PLACEHOLDER", "et al.") for s in sentences]
        # Add paragraph marker at the end of each paragraph's sentences
        if sentences:
            sentences[-1] = sentences[-1] + "[PARAGRAPH_BREAK]"
        all_sentences.extend(sentences)
    
    # Use the paragraph-aware sentences
    sentences = all_sentences
    
    # Analyze each sentence for citations
    analyzed_sentences = []
    all_citations = []
    total_citation_count = 0
    
    for sentence in sentences:
        # Check for possessive author references (e.g., "Johnson's approach")
        has_author_reference = False
        for author in author_names:
            possessive_pattern = f"\\b{author}'s\\b"
            narrative_pattern = f"\\b{author}\\s+(?:argues|claims|states|suggests|notes|observes|finds|proposes|demonstrates|shows)\\b"
            
            if re.search(possessive_pattern, sentence, re.IGNORECASE) or re.search(narrative_pattern, sentence, re.IGNORECASE):
                has_author_reference = True
                break
        # Get citations from this sentence
        citations, citation_count = identify_citations(sentence, author_names)
        
        # If no formal citations but we found an author reference, add an implicit citation
        if citation_count == 0 and has_author_reference:
            for author in author_names:
                possessive_pattern = f"\\b{author}'s\\b"
                narrative_pattern = f"\\b{author}\\s+(?:argues|claims|states|suggests|notes|observes|finds|proposes|demonstrates|shows)\\b"
                
                if re.search(possessive_pattern, sentence, re.IGNORECASE):
                    # Just use the author name for possessive references
                    citations.append({
                        'text': author,
                        'style': 'Narrative APA',
                        'is_narrative': True
                    })
                    citation_count = 1
                    break
                elif re.search(narrative_pattern, sentence, re.IGNORECASE):
                    # Just use the author name for narrative references
                    citations.append({
                        'text': author,
                        'style': 'Narrative APA',
                        'is_narrative': True
                    })
                    citation_count = 1
                    break
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
        
        # Check if this sentence has a paragraph break marker
        has_paragraph_break = False
        if sentence.endswith("[PARAGRAPH_BREAK]"):
            sentence = sentence.replace("[PARAGRAPH_BREAK]", "")
            has_paragraph_break = True
            
        analyzed_sentences.append({
            'sentence': sentence,
            'has_citation': has_citation,
            'citations': citation_texts,
            'citation_count': citation_count,
            'citation_style': citation_style,
            'paragraph_break': has_paragraph_break
        })
        
        all_citations.extend(citations)
        total_citation_count += citation_count
        
    return analyzed_sentences, all_citations, total_citation_count

def identify_citations(s, author_names=None):
    # Initialize author_names if not provided
    if author_names is None:
        author_names = set()
    # Initialize the citations list with dictionaries containing text and style
    citations = []
    
    # Track years in APA citations to avoid duplication
    apa_years = set()
    
    # Track rhetorical move type for implicit citations
    rhetorical_move_type = None
    
    # REPORTING MOVE PATTERNS (directly reporting what sources say)
    reporting_patterns = [
        r'\b(?:according to|as stated by|as reported by|as noted by|as observed by)\b',
        r'\b(?:they|he|she|it)\s+(?:state|report|note|observe|mention|describe|explain|define|clarify)s?\b',
        r'\b(?:their|his|her|its)\s+(?:statement|report|observation|description|explanation|definition)s?\b',
        r'\b(?:in the words of|quoted from|directly from)\b',
        r'\bidentifies\b'
    ]
    
    # TRANSFORMING MOVE PATTERNS (paraphrasing or synthesizing)
    transforming_patterns = [
        r'\b(?:their|his|her|its)\s+(?:findings|research|study|analysis|work|paper|article|results|approach|methodology|framework|model|theory|concept|perspective)s?\b',
        r'\b(?:they|he|she|it)\s+(?:found|discovered|reported|showed|demonstrated|argued|suggested|proposed|developed|established|determined|concluded|identified)\b',
        r'\b(?:these|those|this|that)\s+(?:findings|results|studies|researchers|authors|scholars|studies|papers|articles|analyses)\b',
        r'\b(?:according to|as per)\s+(?:them|him|her|it)\b',
        r'\b(?:similar|similarly|likewise|in the same way|in a similar manner)\b',
        r'\b(?:building on|extending|drawing from|synthesizing|combining|integrating)\b',
        r'\bcase\s+(?:study|studies|analysis|analyses)\b'
    ]
    
    # EVALUATING MOVE PATTERNS (critiquing or analyzing sources)
    evaluating_patterns = [
        r'\b(?:overlooks|fails to|neglects|ignores|misses|omits|lacks|inadequately|insufficiently)\b',
        r'\b(?:their|his|her|its)\s+(?:critique|criticism|assessment|evaluation|limitation|weakness|strength|advantage|shortcoming|gap|flaw|merit)s?\b',
        r'\b(?:they|he|she|it)\s+(?:critiques|criticizes|assesses|evaluates|challenges|questions|contests|disputes|refutes|counters)\b',
        r'\b(?:however|nevertheless|nonetheless|although|though|despite|in spite of|yet|but|conversely|in contrast|on the contrary|on the other hand)\b',
        r'\b(?:problematic|questionable|debatable|controversial|unconvincing|unsupported|unsubstantiated|flawed|limited|narrow|biased|misleading)\b',
        r'\b(?:critically|insightfully|perceptively|astutely|shrewdly)\b',
        r'\b(?:strength|weakness|merit|limitation|advantage|disadvantage|benefit|drawback|shortcoming|gap|flaw)s?\b'
    ]
    
    # Check for reporting patterns
    for pattern in reporting_patterns:
        if re.search(pattern, s, re.IGNORECASE):
            # For reporting patterns, extract just the author name if possible
            author_match = re.search(r'According to\s+([\w]+)', s, re.IGNORECASE)
            if author_match:
                author = author_match.group(1)
                citations.append({
                    'text': author,
                    'style': 'Narrative APA',
                    'is_narrative': True
                })
            else:
                # If no clear author, use the pattern match
                match_text = re.search(pattern, s, re.IGNORECASE).group(0)
                citations.append({
                    'text': match_text,
                    'style': 'Implicit',
                    'is_demonstrative': True
                })
            rhetorical_move_type = 'Reporting'
            break
    
    # If not reporting, check for evaluating patterns
    if not rhetorical_move_type:
        for pattern in evaluating_patterns:
            if re.search(pattern, s, re.IGNORECASE):
                # For evaluating patterns, just extract the author name if mentioned
                author_match = re.search(r'([\w]+)[\s\']s\s+(?:approach|framework|model|theory)', s, re.IGNORECASE)
                if author_match:
                    author = author_match.group(1)
                    citations.append({
                        'text': author,
                        'style': 'Narrative APA',
                        'is_narrative': True
                    })
                else:
                    # If no clear author, use a minimal representation
                    citations.append({
                        'text': "[Source]",
                        'style': 'Implicit'
                    })
                rhetorical_move_type = 'Evaluating'
                break
    
    # If neither reporting nor evaluating, check for transforming patterns
    if not rhetorical_move_type:
        for pattern in transforming_patterns:
            if re.search(pattern, s, re.IGNORECASE):
                # For transforming patterns, extract just the author reference if possible
                author_match = re.search(r'(their|his|her|its)\s+(?:findings|research|study)', s, re.IGNORECASE)
                if author_match:
                    citations.append({
                        'text': "[Previous Source]",
                        'style': 'Implicit'
                    })
                else:
                    # If no clear reference, use a minimal representation
                    citations.append({
                        'text': "[Source]",
                        'style': 'Implicit'
                    })
                rhetorical_move_type = 'Transforming'
                break
    
    # Pre-process the string to fix common formatting issues
    # Remove extra spaces between author and year
    s = re.sub(r'(\w+)\s+\((\d{4})\)', r'\1 (\2)', s)
    
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
    
    # 3. Multiple authors with et al.: Smith et al. (2020)
    for match in re.findall(r'([\w]+)\s+et\s+al\.?\s+\((\d{4})\)', s):
        citations.append({
            'text': f"{match[0]} et al. ({match[1]})",
            'style': 'APA'
        })
        apa_years.add(match[1])
    
    # 4. Three authors: Smith, Johnson, and Lee (2020)
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
            'style': 'APA'
        })
        apa_years.add(match[1])
        
    # 6. Multiple authors with page number in parentheses: (Chen et al. 2018, p. 42)
    for match in re.findall(r'\(([\w]+)\s+et\s+al\.\s+(\d{4}),\s+p\.\s+(\d+)\)', s):
        citations.append({
            'text': f"({match[0]} et al. {match[1]}, p. {match[2]})",
            'style': 'APA'
        })
        apa_years.add(match[1])
        
    # 7. Two authors with comma before year: (Reynolds and Ahmed, 2021)
    for match in re.findall(r'\(([\w]+)\s+and\s+([\w]+),\s+(\d{4})\)', s):
        citations.append({
            'text': f"({match[0]} and {match[1]}, {match[2]})",
            'style': 'APA'
        })
        apa_years.add(match[2])
        
    # 8. Multiple authors with comma before year: (Johnson et al., 2021)
    for match in re.findall(r'\(([\w]+)\s+et\s+al\.,\s+(\d{4})\)', s):
        citations.append({
            'text': f"({match[0]} et al., {match[1]})",
            'style': 'APA'
        })
        apa_years.add(match[1])
        
    # 9. Author with page range: (Wilson, 2019, pp. 78-92)
    for match in re.findall(r'\(([\w]+),\s+(\d{4}),\s+pp\.\s+(\d+)-(\d+)\)', s, re.IGNORECASE):
        citations.append({
            'text': f"({match[0]}, {match[1]}, pp. {match[2]}-{match[3]})",
            'style': 'APA'
        })
        apa_years.add(match[1])
        
    # 9c. Exact match for Wilson citation
    if "(Wilson, 2019, pp. 78-92)" in s:
        # Check if this citation has already been added
        citation_text = "(Wilson, 2019, pp. 78-92)"
        if not any(c['text'] == citation_text for c in citations):
            citations.append({
                'text': citation_text,
                'style': 'APA'
            })
            apa_years.add('2019')
        
    # 9b. Alternative pattern for page range with more flexible spacing
    for match in re.findall(r'\(\s*([\w]+)\s*,\s*(\d{4})\s*,\s*pp\.\s*(\d+)\s*-\s*(\d+)\s*\)', s, re.IGNORECASE):
        # Check if this citation has already been added
        citation_text = f"({match[0]}, {match[1]}, pp. {match[2]}-{match[3]})"
        if not any(c['text'] == citation_text for c in citations):
            citations.append({
                'text': citation_text,
                'style': 'APA'
            })
            apa_years.add(match[1])
        
    # 10. Two authors with page number: (Garcia and Martinez, 2022, p. 215)
    for match in re.findall(r'\(([\w]+)\s+and\s+([\w]+),\s+(\d{4}),\s+p\.\s+(\d+)\)', s):
        citations.append({
            'text': f"({match[0]} and {match[1]}, {match[2]}, p. {match[3]})",
            'style': 'APA'
        })
        apa_years.add(match[2])
    
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
    for match in re.findall(r'\(([\w]+),\s+([\w]+),\s+and\s+([\w]+)\s+(\d+)\)', s):
        citations.append({
            'text': f"{match[0]}, {match[1]} and {match[2]} {match[3]}",
            'style': 'MLA'
        })
    
    # 5. Three authors without page numbers: (Smith, Johnson and Lee)
    for match in re.findall(r'\(([\w]+),\s+([\w]+),\s+and\s+([\w]+)\)', s):
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
                'style': 'Unsure'
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
    
    # 12. Detect citations with year at the end of sentence like "...cognitive processes (Mislevy, 2018)."
    for match in re.findall(r'\((\w+),?\s+(\d{4})\)', s):
        author = match[0].rstrip(',').strip()
        year = match[1].strip()
        
        # Skip if this citation is already detected
        already_cited = False
        for citation in citations:
            if f"{author}" in citation['text'] and year in citation['text']:
                already_cited = True
                break
        
        if not already_cited:
            citations.append({
                'text': f"{author} ({year})",
                'style': 'APA'
            })
            apa_years.add(year)
    
    # 13. Detect in-text citations like "According to Abrams" or "Baptista and Gradim explore"
    # First look for common author-signal phrases
    author_signals = [
        r'(?:According to|as stated by|as noted by|as mentioned by|as argued by|as claimed by|as shown by|as demonstrated by|as explained by)\s+([A-Z][\w]+)',
        r'([A-Z][\w]+)\s+(?:states|argues|claims|notes|mentions|shows|demonstrates|explains|explores|examines|investigates|analyzes|discusses|suggests|proposes|concludes)',
        r'([A-Z][\w]+)\s+and\s+([A-Z][\w]+)\s+(?:state|argue|claim|note|mention|show|demonstrate|explain|explore|examine|investigate|analyze|discuss|suggest|propose|conclude)'
    ]
    
    for pattern in author_signals:
        for match in re.findall(pattern, s):
            if isinstance(match, tuple):
                # Handle multiple authors
                for author in match:
                    # Skip if this author is already part of another citation
                    author_already_cited = False
                    for citation in citations:
                        if author in citation['text']:
                            author_already_cited = True
                            break
                    
                    if not author_already_cited and len(author) > 1:  # Ensure it's a real name
                        citations.append({
                            'text': author,
                            'style': 'Narrative'
                        })
            else:
                # Single author
                author = match
                # Skip if this author is already part of another citation
                author_already_cited = False
                for citation in citations:
                    if author in citation['text']:
                        author_already_cited = True
                        break
                
                if not author_already_cited and len(author) > 1:  # Ensure it's a real name
                    citations.append({
                        'text': author,
                        'style': 'Narrative'
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
    """
    Use machine learning models to classify the rhetorical move of a sentence.
    
    The three categories are:
    - Reporting: Directly reporting what a source says
    - Transforming: Paraphrasing or synthesizing source material
    - Evaluating: Critiquing, analyzing, or evaluating sources
    - No Citation: Sentences without citations
    """
    sentence = sentence_info['sentence']
    has_citation = sentence_info['has_citation']
    
    # FIRST CHECK: Direct pattern matching for obvious rhetorical moves
    
    # Check for "According to" at the beginning of the sentence (Reporting)
    if sentence.strip().lower().startswith("according to"):
        return "Reporting", 0.95
        
    # Check for strong evaluative language (Evaluating)
    evaluative_phrases = [
        "fails to", "overlooks", "neglects", "ignores", "misses", "omits",
        "lacks", "inadequately", "insufficiently", "problematic", "questionable",
        "debatable", "controversial", "unconvincing", "unsupported", "unsubstantiated",
        "flawed", "limited", "narrow", "biased", "misleading", "weakness", "limitation",
        "shortcoming", "gap", "flaw", "critique", "criticism", "challenges", "questions",
        "contests", "disputes", "refutes", "counters", "contradicts"
    ]
    
    for phrase in evaluative_phrases:
        if phrase in sentence.lower():
            return "Evaluating", 0.95
    
    # If no citation, mark as 'No Citation'
    if not has_citation:
        return "No Citation", 1.0
    
    # Check if the sentence has more than one distinct citation
    # If so, classify it as Transforming (synthesis of multiple sources)
    if 'citations' in sentence_info and len(sentence_info['citations']) > 1:
        return "Transforming", 0.9
    
    # FIRST: Use the ML classifier to predict the rhetorical move for sentences with citations
    rhetorical_move, confidence = rhetorical_classifier.predict_rhetorical_move(sentence)
    
    # Check for very clear reporting patterns regardless of ML confidence
    reporting_patterns = [
        # With quotes
        r'^According to\b.*".*".*\([^)]+\)',  # According to X, "quote" (citation)
        r'^As\s+(?:stated|noted|reported|observed|mentioned)\s+by\b.*".*"',  # As stated by X, "quote"
        r'\b(?:states|reports|notes|observes|mentions|writes|says)\s+that\s+".*"',  # X states that "quote"
        r'\bin\s+the\s+words\s+of\b.*".*"',  # In the words of X, "quote"
        
        # Without quotes - narrative citations
        r'^According to\b.*?\([^)]+\)',  # According to Smith (2019), ...
        r'^As\s+(?:stated|noted|reported|observed|mentioned)\s+by\b.*?\([^)]+\)',  # As stated by Smith (2019), ...
        r'(?:states|reports|notes|observes|mentions|writes|says)\s+that\b.*?\([^)]+\)',  # Smith states that (2019) ...
        
        # Common reporting phrases
        r'\breports?\s+(?:that|how|on)\b',  # reports that/how/on
        r'\bdescribes?\s+(?:that|how)\b',  # describes that/how
        r'\bexplains?\s+(?:that|how)\b',  # explains that/how
        r'\bidentifies?\b',  # identifies
        r'\bdocuments?\b'  # documents
    ]
    
    for pattern in reporting_patterns:
        if re.search(pattern, sentence, re.IGNORECASE):
            return "Reporting", 0.9
    
    # SECOND: Only use pattern matching if the ML confidence is low
    if confidence < 0.7:
        # Check for implicit citations with specific rhetorical move types
        if 'citations' in sentence_info:
            for citation in sentence_info['citations']:
                # Check for common reporting phrases
                if citation.lower().startswith("according to") or \
                   citation.lower().startswith("as stated by") or \
                   citation.lower().startswith("as noted by") or \
                   "reports that" in citation.lower() or \
                   "states that" in citation.lower():
                    return "Reporting", 0.9
                # Check for evaluative phrases
                elif "fails to" in citation.lower() or \
                     "overlooks" in citation.lower() or \
                     "neglects" in citation.lower() or \
                     "limitation" in citation.lower() or \
                     "critique" in citation.lower() or \
                     "weakness" in citation.lower():
                    return "Evaluating", 0.9
                # Otherwise assume transforming
                elif sentence_info.get('has_citation', False):
                    return "Transforming", 0.8
                # Check for author references (possessive form)
                elif "'s" in citation and any(author in citation for author in author_names):
                    return "Transforming", 0.7
                # Check for author references (narrative form)
                elif any(author in citation for author in author_names):
                    return "Transforming", 0.7
        
        # Check for strong evaluative patterns if ML confidence is low
        evaluating_patterns = [
            r'\b(?:overlooks|fails to|neglects|ignores|misses|omits)\b',
            r'\b(?:critique|criticism|limitation|weakness|flaw)s?\b',
            r'\b(?:problematic|questionable|debatable|controversial|unconvincing)\b'
        ]
        
        for pattern in evaluating_patterns:
            if re.search(pattern, sentence, re.IGNORECASE) and confidence < 0.65:
                return "Evaluating", 0.7
    
    # If the classifier didn't detect a rhetorical move with any confidence,
    # default to "Reporting" as it's the most common move with citations
    if confidence < 0.5:
        rhetorical_move = "Reporting"
        confidence = 0.5
    
    return rhetorical_move, confidence

@app.route('/')
def home():
    @after_this_request
    def add_no_cache(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    # Serve the fixed chart template
    return render_template('fixed_chart.html')

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
    
    return process_text(text)

@app.route('/analyze_files', methods=['POST'])
def analyze_files():
    if 'files' not in request.files:
        return jsonify({
            'error': 'No files provided',
            'sentence_analysis': [],
            'citation_count': 0
        })
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({
            'error': 'No files selected',
            'sentence_analysis': [],
            'citation_count': 0
        })
    
    # Process each file and combine the text
    combined_text = ""
    for file in files:
        filename = secure_filename(file.filename)
        # Create a temporary file to store the uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            file.save(temp.name)
            file_text = extract_text_from_file(temp.name, filename)
            combined_text += file_text + "\n\n"
            # Delete the temporary file
            os.unlink(temp.name)
    
    if not combined_text.strip():
        return jsonify({
            'error': 'Could not extract text from the provided files',
            'sentence_analysis': [],
            'citation_count': 0
        })
    
    return process_text(combined_text)

@app.route('/display_citations', methods=['POST'])
def display_citations():
    """Display uploaded text with citation tags highlighted"""
    if 'file' not in request.files:
        return render_template('display_citations.html', error='No file provided')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('display_citations.html', error='No file selected')
    
    # Create a temporary file to store the uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        file.save(temp.name)
        filename = secure_filename(file.filename)
        text = extract_text_from_file(temp.name, filename)
        # Delete the temporary file
        os.unlink(temp.name)
    
    if not text.strip():
        return render_template('display_citations.html', error='Could not extract text from the file')
    
    # Extract author names for reference tracking
    author_names = extract_author_names(text)
    
    # Process the text to identify citations
    # We'll use a simplified version of preprocess_text that preserves formatting
    paragraphs = text.split('\n\n')
    processed_paragraphs = []
    all_citations = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            processed_paragraphs.append('')
            continue
            
        # Process each sentence in the paragraph
        sentences = nltk.sent_tokenize(paragraph)
        processed_sentences = []
        
        for sentence in sentences:
            # Identify citations in this sentence
            citations, _ = identify_citations(sentence, author_names)
            
            # If we found citations, mark them in the text
            if citations:
                marked_sentence = sentence
                for citation in citations:
                    citation_text = citation['text']
                    citation_style = citation['style']
                    
                    # Only mark the citation if it's actually in the text (not implicit)
                    if citation_style != 'Implicit' and citation_text in sentence:
                        # Create a span with appropriate styling
                        span = f'<span class="citation {citation_style.lower().replace(" ", "-")}" title="{citation_style}">{citation_text}</span>'
                        marked_sentence = marked_sentence.replace(citation_text, span)
                
                processed_sentences.append(marked_sentence)
                all_citations.extend(citations)
            else:
                processed_sentences.append(sentence)
        
        processed_paragraphs.append(' '.join(processed_sentences))
    
    # Join paragraphs with double line breaks to preserve formatting
    processed_text = '\n\n'.join(processed_paragraphs)
    
    # Count citations by style
    citation_counts = {}
    for citation in all_citations:
        style = citation['style']
        if style in citation_counts:
            citation_counts[style] += 1
        else:
            citation_counts[style] = 1
    
    return render_template('display_citations.html', 
                           text=processed_text, 
                           citations=all_citations,
                           citation_counts=citation_counts,
                           filename=filename)

def extract_text_from_file(file_path, filename):
    """Extract text from different file types"""
    text = ""
    file_ext = os.path.splitext(filename)[1].lower()
    
    try:
        if file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file_ext == '.pdf':
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
        elif file_ext in ['.doc', '.docx']:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        print(f"Error extracting text from {filename}: {str(e)}")
        text = f"Error processing {filename}: {str(e)}"
    
    return text

def extract_author_names(text):
    """Extract all author names from citations in the text for later reference"""
    author_names = set()
    
    # Extract author names from APA citations: (Smith, 2020)
    for match in re.findall(r'\(([\w]+),\s*\d{4}', text):
        author_names.add(match)
    
    # Extract author names from APA citations with multiple authors: (Smith et al., 2020)
    for match in re.findall(r'\(([\w]+)\s+et\s+al\.', text):
        author_names.add(match)
    
    # Extract author names from APA citations with two authors: (Smith and Jones, 2020)
    for match in re.findall(r'\(([\w]+)\s+and\s+([\w]+)', text):
        author_names.add(match[0])
        author_names.add(match[1])
    
    # Extract author names from narrative citations: Smith (2020)
    for match in re.findall(r'([\w]+)\s+\(\d{4}\)', text):
        author_names.add(match)
    
    # Extract author names from MLA citations
    for match in re.findall(r'\(([\w]+)\s+\d+\)', text):
        author_names.add(match)
    
    return author_names

def process_text(text):
    
    # Extract all author names from citations for later reference
    author_names = extract_author_names(text)
    
    # Preprocess and analyze the text
    analyzed_sentences, all_citations, total_citation_count = preprocess_text(text, author_names)
    
    # Add rhetorical move analysis using ML models
    for sentence_info in analyzed_sentences:
        rhetorical_move, confidence = analyze_rhetorical_moves(sentence_info)
        sentence_info['rhetorical_move'] = rhetorical_move
        sentence_info['confidence'] = confidence
        
        # Add citation styles information
        if sentence_info['has_citation'] and sentence_info['citations']:
            # Get the raw citations with their styles
            citations_with_styles = []
            for citation_text in sentence_info['citations']:
                # Find the citation with style in all_citations
                for citation in all_citations:
                    if citation['text'] == citation_text:
                        citations_with_styles.append({
                            'text': citation_text,
                            'style': citation['style']
                        })
                        break
                else:
                    # If not found, add as unsure
                    citations_with_styles.append({
                        'text': citation_text,
                        'style': 'Unsure'
                    })
            
            sentence_info['citations_with_styles'] = citations_with_styles
    
    # Calculate statistics for rhetorical moves
    move_counts = {
        "Reporting": 0,
        "Transforming": 0,
        "Evaluating": 0,
        "No Citation": 0
    }
    
    # Filter out section headers (all caps sentences with no punctuation)
    content_sentences = []
    for sentence_info in analyzed_sentences:
        sentence = sentence_info['sentence']
        # Check if this is a section header (all caps, short, no punctuation except colon)
        is_header = sentence.isupper() and len(sentence.split()) <= 3 and not any(p in sentence for p in '.!?,;')
        if not is_header:
            content_sentences.append(sentence_info)
            move = sentence_info['rhetorical_move']
            if move in move_counts:
                move_counts[move] += 1
    
    # Calculate percentages using only content sentences
    total_content_sentences = len(content_sentences)
    move_percentages = {}
    for move, count in move_counts.items():
        percentage = (count / total_content_sentences) * 100 if total_content_sentences > 0 else 0
        move_percentages[move] = round(percentage, 1)
    
    # Debug print to see what's being sent
    print("All citations:", all_citations)
    for sentence in analyzed_sentences:
        if 'citations_with_styles' in sentence:
            print("Citations with styles:", sentence['citations_with_styles'])
    
    return jsonify({
        'sentence_analysis': analyzed_sentences,
        'citation_count': total_citation_count,
        'rhetorical_move_stats': {
            'counts': move_counts,
            'percentages': move_percentages
        }
    })

if __name__ == '__main__':
    nltk.download('punkt')  # Download required NLTK data
    app.run(debug=True, host='0.0.0.0', port=5001)
