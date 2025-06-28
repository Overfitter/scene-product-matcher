"""
Vocabularies and categorization systems
"""

class Vocabularies:
    """All vocabulary definitions"""
    
    @staticmethod
    def get_room_prompts():
        return [
            "elegant contemporary living room with modern sectional sofa, neutral colors, and sophisticated styling",
            "luxurious bedroom with premium bedding, sophisticated furniture, and refined decor",
            "sophisticated dining room with elegant table setting and premium furnishings",
            "modern kitchen with clean lines, premium appliances, and contemporary design",
            "spa-like bathroom with clean lines, premium fixtures, and luxury materials",
            "professional home office with organized workspace and modern furniture",
            "welcoming entryway with stylish console and carefully curated decor",
            "refined hallway with artwork, elegant lighting, and sophisticated styling"
        ]
    
    @staticmethod
    def get_room_types():
        return [
            "living room", "bedroom", "dining room", "kitchen",
            "bathroom", "office", "entryway", "hallway"
        ]
    
    @staticmethod
    def get_style_prompts():
        return [
            "contemporary modern interior featuring clean lines, neutral palette, geometric forms, and sophisticated minimalist furniture",
            "traditional classic interior with elegant wood furniture, warm colors, ornate details, and timeless sophisticated design",
            "pure minimalist interior with sparse white walls, essential furniture only, clean lines, and maximum negative space",
            "transitional interior seamlessly blending contemporary and traditional elements with balanced sophisticated styling",
            "luxury sophisticated interior featuring premium materials, elegant finishes, high-end furnishings, and refined details",
            "rustic vintage interior with weathered wood, antique pieces, farmhouse charm, and authentic vintage character",
            "bohemian eclectic interior with vibrant patterns, mixed textures, artistic elements, and creative layered styling"
        ]
    
    @staticmethod
    def get_design_styles():
        return [
            "contemporary", "traditional", "minimalist", "transitional",
            "luxury", "rustic", "bohemian"
        ]
    
    @staticmethod
    def get_color_system():
        return {
            'neutral_warm': {
                'colors': ['cream', 'beige', 'warm gray', 'taupe', 'ivory', 'off-white'],
                'description': 'warm neutral tones with creams and beiges',
                'harmony_score': 1.2
            },
            'neutral_cool': {
                'colors': ['cool gray', 'white', 'silver', 'pearl', 'platinum'],
                'description': 'cool neutral tones with grays and whites',
                'harmony_score': 1.2
            },
            'warm_metallics': {
                'colors': ['gold', 'brass', 'copper', 'bronze', 'amber'],
                'description': 'warm metallic accents and finishes',
                'harmony_score': 1.1
            },
            'cool_metallics': {
                'colors': ['silver', 'chrome', 'platinum', 'steel'],
                'description': 'cool metallic accents and finishes',
                'harmony_score': 1.1
            },
            'earth_tones': {
                'colors': ['brown', 'tan', 'rust', 'terracotta', 'natural'],
                'description': 'natural earth tone palette',
                'harmony_score': 1.0
            },
            'blues': {
                'colors': ['navy', 'blue', 'teal', 'indigo', 'sapphire'],
                'description': 'sophisticated blue color family',
                'harmony_score': 1.0
            },
            'greens': {
                'colors': ['sage', 'green', 'emerald', 'forest', 'mint'],
                'description': 'natural green color palette',
                'harmony_score': 1.0
            }
        }
    
    @staticmethod
    def get_product_categories():
        return {
            'statement_vases': {
                'keywords': ['vase', 'urn', 'large vase', 'floor vase'],
                'size_preference': ['large', 'medium'],
                'materials': ['ceramic', 'glass', 'metal'],
                'placement_priority': 1.0,
                'style_alignment': {
                    'contemporary': ['geometric', 'modern', 'clean', 'simple', 'sculptural'],
                    'traditional': ['classic', 'elegant', 'ornate', 'detailed', 'refined'],
                    'minimalist': ['simple', 'clean', 'minimal', 'pure', 'essential']
                },
                'placement': 'statement piece on coffee table, console, or floor as focal point'
            },
            'accent_vases': {
                'keywords': ['small vase', 'bud vase', 'mini vase'],
                'size_preference': ['small'],
                'materials': ['ceramic', 'glass'],
                'placement_priority': 0.8,
                'style_alignment': {
                    'contemporary': ['modern', 'clean', 'geometric'],
                    'traditional': ['delicate', 'elegant', 'refined']
                },
                'placement': 'grouped on side tables or shelving for subtle accent'
            },
            'sculptural_objects': {
                'keywords': ['sculpture', 'figurine', 'elephant', 'animal', 'bird', 'abstract', 'art piece'],
                'size_preference': ['small', 'medium'],
                'materials': ['ceramic', 'metal', 'resin', 'wood'],
                'placement_priority': 0.9,
                'style_alignment': {
                    'contemporary': ['abstract', 'geometric', 'modern', 'sculptural'],
                    'traditional': ['classic', 'detailed', 'ornate', 'realistic'],
                    'minimalist': ['simple', 'clean', 'essential']
                },
                'placement': 'shelving, console, or coffee table as artistic focal point'
            },
            'lighting_accents': {
                'keywords': ['candle', 'candleholder', 'candlestick', 'lantern', 'light'],
                'size_preference': ['small', 'medium'],
                'materials': ['metal', 'glass', 'ceramic'],
                'placement_priority': 0.9,
                'style_alignment': {
                    'contemporary': ['modern', 'clean', 'geometric', 'minimal'],
                    'traditional': ['classic', 'ornate', 'elegant'],
                    'luxury': ['crystal', 'premium', 'sophisticated']
                },
                'placement': 'ambient lighting on coffee table, console, or dining table'
            },
            'functional_beauty': {
                'keywords': ['bowl', 'tray', 'dish', 'serving', 'decorative bowl'],
                'size_preference': ['small', 'medium'],
                'materials': ['ceramic', 'wood', 'metal', 'glass'],
                'placement_priority': 0.8,
                'style_alignment': {
                    'contemporary': ['modern', 'sleek', 'geometric', 'clean'],
                    'traditional': ['classic', 'elegant', 'refined']
                },
                'placement': 'coffee table styling or console display with functional appeal'
            },
            'storage_style': {
                'keywords': ['jar', 'box', 'container', 'chest', 'canister'],
                'size_preference': ['small', 'medium'],
                'materials': ['ceramic', 'wood', 'metal'],
                'placement_priority': 0.7,
                'style_alignment': {
                    'contemporary': ['clean', 'modern', 'minimal'],
                    'traditional': ['classic', 'detailed', 'elegant']
                },
                'placement': 'stylish storage solutions for coffee table or console'
            },
            'accent_tables': {
                'keywords': ['table', 'stool', 'stand', 'pedestal', 'accent table'],
                'size_preference': ['medium', 'large'],
                'materials': ['wood', 'metal', 'glass'],
                'placement_priority': 1.0,
                'style_alignment': {
                    'contemporary': ['modern', 'clean', 'metal', 'glass', 'geometric'],
                    'traditional': ['wood', 'classic', 'elegant', 'detailed']
                },
                'placement': 'functional accent beside seating or as plant stand'
            },
            'decorative_accents': {
                'keywords': ['decorative', 'accent', 'ornament', 'decoration'],
                'size_preference': ['small', 'medium'],
                'materials': ['ceramic', 'metal', 'glass', 'wood'],
                'placement_priority': 0.6,
                'style_alignment': {
                    'contemporary': ['abstract', 'geometric', 'modern'],
                    'traditional': ['classic', 'detailed', 'ornate']
                },
                'placement': 'accent pieces for shelving, console, or side tables'
            }
        }