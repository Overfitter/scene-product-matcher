"""
Description Processing Utilities for Ultimate Matcher
Simplified version focusing on essential preprocessing only
"""

import re
from typing import Dict, List

class DescriptionProcessor:
    """Lightweight description processor for the Ultimate Matcher"""
    
    def __init__(self):
        # Essential pattern replacements
        self.patterns = {
            r'\b(\d+)X(\d+)X(\d+)\b': r'\1 by \2 by \3 inch',
            r'\b(\d+)"\b': r'\1 inch', 
            r'\b(\d+)\s*H\b': r'\1 inch height',
            r'\b(\d+)\s*W\b': r'\1 inch width',
            r'\b(\d+)\s*D\b': r'\1 inch depth',
            r'\b2/A\b': 'set of 2',
            r'\bS/2\b': 'set of 2',
            r'\bS/3\b': 'set of 3',
            r'\bSET\s+OF\s+(\d+)\b': r'set of \1',
            r'\bPAIR\b': 'set of 2',
        }
        
        # Common abbreviations
        self.abbreviations = {
            'RSN': 'resin',
            'TRI': 'triangular', 
            'GEO': 'geometric',
            'RECT': 'rectangular',
            'SQ': 'square',
            'RND': 'round',
            'OVAL': 'oval',
            'WHT': 'white',
            'BLK': 'black',
            'BLU': 'blue',
            'GRN': 'green',
            'RED': 'red',
            'YEL': 'yellow',
            'SLV': 'silver',
            'GLD': 'gold',
            'BRZ': 'bronze'
        }
    
    def clean_description(self, description: str) -> str:
        """Clean and normalize product description"""
        
        if not description:
            return ""
        
        # Convert to uppercase for processing
        desc = description.upper().strip()
        
        # Apply pattern replacements
        for pattern, replacement in self.patterns.items():
            desc = re.sub(pattern, replacement, desc, flags=re.IGNORECASE)
        
        # Expand abbreviations
        words = desc.split()
        expanded_words = []
        for word in words:
            # Remove punctuation for abbreviation lookup
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.abbreviations:
                expanded_words.append(self.abbreviations[clean_word])
            else:
                expanded_words.append(word)
        
        desc = ' '.join(expanded_words)
        
        # Clean up formatting
        desc = re.sub(r'[,\-/]+', ' ', desc)
        desc = re.sub(r'\s+', ' ', desc).strip()
        
        return desc.lower()
    
    def extract_basic_features(self, description: str) -> Dict:
        """Extract basic features from description"""
        
        cleaned = self.clean_description(description)
        
        # Extract materials
        materials = re.findall(
            r'\b(ceramic|crystal|marble|stone|metal|brass|copper|silver|gold|wood|wooden|glass|fabric|resin|leather|rattan|wicker)\b',
            cleaned
        )
        
        # Extract colors
        colors = re.findall(
            r'\b(white|black|blue|navy|red|green|sage|brown|gold|brass|silver|gray|grey|yellow|beige|cream|ivory|pearl|bronze|copper|rust|teal|charcoal|natural|clear)\b',
            cleaned
        )
        
        # Extract size numbers
        size_numbers = re.findall(r'(\d+)(?:\s*(?:inch|in|ft|foot|cm|mm))', cleaned)
        
        # Determine size category
        if size_numbers:
            max_size = max([int(n) for n in size_numbers])
            if max_size < 6:
                size_category = 'small'
            elif max_size < 18:
                size_category = 'medium'  
            else:
                size_category = 'large'
        else:
            size_category = 'medium'  # Default
        
        # Check if it's a set
        is_set = bool(re.search(r'\b(set|pair|2/a|s/2|s/3)\b', cleaned))
        
        return {
            'cleaned_description': cleaned,
            'materials': list(set(materials)),
            'colors': list(set(colors)),
            'size_category': size_category,
            'size_numbers': [int(n) for n in size_numbers],
            'is_set': is_set,
            'word_count': len(cleaned.split())
        }