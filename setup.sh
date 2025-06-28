# Complete VS Code Project Setup for Scene-to-Product Matcher
# ============================================================

# 1. CREATE PROJECT DIRECTORY STRUCTURE
mkdir scene-product-matcher
cd scene-product-matcher

# Create main directories
mkdir src
mkdir data
mkdir cache
mkdir tests
mkdir docs
mkdir scripts
mkdir api
mkdir models
mkdir notebooks
mkdir static
mkdir static/css
mkdir static/js
mkdir templates
mkdir logs

# 2. CREATE VIRTUAL ENVIRONMENT
python -m venv venv

# 3. ACTIVATE VIRTUAL ENVIRONMENT
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# 4. CREATE PROJECT FILES
touch README.md
touch requirements.txt
touch .env.example
touch .gitignore
touch Dockerfile
touch docker-compose.yml
touch Makefile

# Core application files
touch src/__init__.py
touch src/scene_matcher.py
touch src/advanced_scene_analysis.py
touch src/catalog_processor.py
touch src/config.py
touch src/utils.py

# API files
touch api/__init__.py
touch api/main.py
touch api/models.py
touch api/routes.py
touch api/dependencies.py

# Test files
touch tests/__init__.py
touch tests/test_scene_matcher.py
touch tests/test_api.py
touch tests/conftest.py

# Script files
touch scripts/setup_catalog.py
touch scripts/build_embeddings.py
touch scripts/demo.py

# Notebook files
touch notebooks/demo_analysis.ipynb
touch notebooks/catalog_exploration.ipynb

# Configuration files
touch .vscode/settings.json
touch .vscode/launch.json
touch .vscode/tasks.json

# Data files (you'll place your CSV here)
touch data/productcatalog.csv
touch data/sample_scenes/living_room.jpg

echo "Project structure created successfully!"
echo "Next: Copy the code files from the artifacts..."