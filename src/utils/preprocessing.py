"""
Description processing utilities
"""

import re
from typing import Dict, List
from collections import defaultdict

class DescriptionProcessor:
    """Description processing methods from original code"""
    
    def __init__(self):
        # Enterprise-grade pattern replacements from original code
        self.patterns = {
            r'\b(\d+)X(\d+)X(\d+)\b': r'\1 by \2 by \3 inch',
            r'\b(\d+)"\b': r'\1 inch',
            r'\b(\d+)\'': r'\1 foot',
            r'\b(\d+)\s*H\b': r'\1 inch height',
            r'\b(\d+)\s*W\b': r'\1 inch width',
            r'\b(\d+)\s*D\b': r'\1 inch depth',
            r'\b2/A\b': 'set of 2',
            r'\bS/2\b': 'set of 2',
            r'\bS/3\b': 'set of 3',
            r'\bSET\s+OF\s+(\d+)\b': r'set of \1',
            r'\bPAIR\b': 'set of 2',
            # Materials with quality indicators
            r'\bCERAMIC\b': 'premium ceramic',
            r'\bMETAL\b': 'quality metal',
            r'\bWOOD\b': 'natural wood',
            r'\bWOODEN\b': 'wooden',
            r'\bGLASS\b': 'glass',
            r'\bCRYSTAL\b': 'premium crystal',
            r'\bMARBLE\b': 'luxury marble',
            r'\bSTONE\b': 'natural stone',
            r'\bRESIN\b': 'resin',
            r'\bFABRIC\b': 'fabric'
        }
    
    def enhanced_description_processing(self, raw_description: str) -> Dict:
        """Ultimate description processing with comprehensive extraction"""
        
        desc = raw_description.upper().strip()
        
        enhanced = desc
        for pattern, replacement in self.patterns.items():
            enhanced = re.sub(pattern, replacement, enhanced, flags=re.IGNORECASE)
        
        # Advanced extraction with quality scoring
        materials = re.findall(
            r'\b(ceramic|crystal|marble|stone|metal|brass|copper|silver|gold|wood|wooden|glass|fabric|resin|leather|rattan|wicker)\b',
            enhanced.lower()
        )
        
        colors = re.findall(
            r'\b(white|black|blue|navy|red|green|sage|brown|gold|brass|silver|gray|grey|yellow|beige|cream|ivory|pearl|bronze|copper|rust|teal|charcoal|off-white|natural|clear)\b',
            enhanced.lower()
        )
        
        # Style descriptors with quality indicators
        style_descriptors = re.findall(
            r'\b(modern|contemporary|traditional|classic|vintage|rustic|elegant|geometric|ornate|simple|clean|minimal|decorative|artistic|sophisticated|luxury|premium|handcrafted)\b',
            enhanced.lower()
        )
        
        # Size analysis with intelligence
        size_numbers = re.findall(r'(\d+)(?:\s*(?:inch|in|ft|foot|cm|mm))', enhanced.lower())
        size_category = self._determine_size_category(size_numbers)
        
        # Set detection
        is_set = bool(re.search(r'\b(set|pair|2/a|s/2|s/3)\b', enhanced.lower()))
        set_count = self._extract_set_count(enhanced)
        
        # Quality scoring based on description richness
        quality_indicators = {
            'has_materials': len(materials) > 0,
            'has_colors': len(colors) > 0,
            'has_style_descriptors': len(style_descriptors) > 0,
            'has_size_info': len(size_numbers) > 0,
            'is_detailed': len(enhanced.split()) > 3,
            'has_premium_materials': any(mat in ['ceramic', 'crystal', 'marble', 'brass', 'copper'] for mat in materials)
        }
        
        quality_score = sum(quality_indicators.values()) / len(quality_indicators)
        
        # Clean formatting
        enhanced = re.sub(r'[,\-/]+', ' ', enhanced)
        enhanced = re.sub(r'\s+', ' ', enhanced).strip().lower()
        
        # Create ultimate contextual description
        context_parts = [enhanced]
        
        if materials:
            unique_materials = list(set(materials))
            if len(unique_materials) == 1:
                context_parts.append(f"expertly crafted from {unique_materials[0]}")
            else:
                context_parts.append(f"beautifully combining {' and '.join(unique_materials[:2])}")
        
        if colors:
            unique_colors = list(set(colors))
            if len(unique_colors) == 1:
                context_parts.append(f"featuring elegant {unique_colors[0]} finish")
            else:
                context_parts.append(f"showcasing sophisticated {' and '.join(unique_colors[:2])} tones")
        
        if style_descriptors:
            unique_styles = list(set(style_descriptors))
            context_parts.append(f"with {' and '.join(unique_styles[:2])} design")
        
        if is_set:
            context_parts.append(f"thoughtfully designed as a coordinated set")
        
        context_parts.append("perfect luxury home accessory for sophisticated interior styling")
        
        contextual_desc = ', '.join(context_parts)
        
        return {
            'enhanced': enhanced.capitalize(),
            'contextual': contextual_desc.capitalize(),
            'materials': list(set(materials)),
            'colors': list(set(colors)),
            'style_descriptors': list(set(style_descriptors)),
            'size_category': size_category,
            'size_numbers': [int(n) for n in size_numbers],
            'is_set': is_set,
            'set_count': set_count,
            'quality_score': quality_score,
            'quality_indicators': quality_indicators,
            'raw_cleaned': enhanced
        }
    
    def _determine_size_category(self, size_numbers: List[str]) -> str:
        """Intelligent size categorization"""
        if not size_numbers:
            return 'medium'
        
        max_size = max([int(n) for n in size_numbers])
        
        if max_size < 6:
            return 'small'
        elif max_size < 18:
            return 'medium'
        else:
            return 'large'
    
    def _extract_set_count(self, description: str) -> int:
        """Extract set count from description"""
        set_patterns = [
            (r'\bset of (\d+)\b', lambda m: int(m.group(1))),
            (r'\bs/(\d+)\b', lambda m: int(m.group(1))),
            (r'\b(\d+)/a\b', lambda m: int(m.group(1))),
            (r'\bpair\b', lambda m: 2)
        ]
        
        for pattern, extractor in set_patterns:
            match = re.search(pattern, description.lower())
            if match:
                return extractor(match)
        
        return 1
    
    def enhanced_categorization(self, description_data: Dict, product_categories: Dict) -> str:
        """Ultimate product categorization with advanced logic"""
        
        desc_lower = description_data['raw_cleaned']
        materials = description_data.get('materials', [])
        size_category = description_data.get('size_category', 'medium')
        quality_score = description_data.get('quality_score', 0.5)
        
        category_scores = defaultdict(float)
        
        # Score each category with advanced logic
        for category, config in product_categories.items():
            score = 0
            
            # Keyword matching with position weighting
            for keyword in config['keywords']:
                if keyword in desc_lower:
                    words = desc_lower.split()
                    if words and keyword == words[0]:
                        score += 5  # First word bonus
                    elif len(words) > 1 and keyword in words[:2]:
                        score += 3  # Early position bonus
                    else:
                        score += 2  # General match
            
            # Size preference alignment
            if size_category in config.get('size_preference', []):
                score += 2
            
            # Material preference alignment
            material_matches = set(materials) & set(config.get('materials', []))
            score += len(material_matches) * 1.5
            
            # Quality bonus
            score += quality_score * config.get('placement_priority', 0.5)
            
            if score > 0:
                category_scores[category] = score
        
        # Return best match or intelligent default
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])[0]
            return best_category
        else:
            # Intelligent fallback based on description patterns
            if any(word in desc_lower for word in ['vase', 'urn']):
                return 'statement_vases'
            elif any(word in desc_lower for word in ['candle', 'light']):
                return 'lighting_accents'
            elif any(word in desc_lower for word in ['bowl', 'dish']):
                return 'functional_beauty'
            else:
                return 'decorative_accents'