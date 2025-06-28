
"""
Entry point for the Ultimate Scene Matcher
Run your original code exactly as written
"""

from core.matcher import SceneProductMatcher

# Your original execution code
matcher = SceneProductMatcher()
matcher.load_and_process_catalog(catalog_path="data/product-catalog.csv")
matcher.build_embeddings_sync()
result = matcher.get_ultimate_recommendations(scene_image_path="data/example_scene.webp", k=10)
print(result)