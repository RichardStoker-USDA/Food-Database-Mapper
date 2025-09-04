---
title: Food Database Mapper
emoji: 🥗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: cc0-1.0
models:
  - thenlper/gte-large
tags:
  - food-science
  - semantic-search
  - usda
  - nutrition
  - gradio
  - python
---

# Food Database Mapper

A sophisticated web application for mapping food descriptions to reference databases using multiple matching algorithms including semantic embeddings, fuzzy matching, and TF-IDF similarity.

## Repository & Hosting

- **Source Code**: <a href="https://github.com/RichardStoker-USDA/Food-Database-Mapper" target="_blank">GitHub Repository</a> - Primary location for code and documentation
- **Live Application**: <a href="https://huggingface.co/spaces/richtext/Food-Database-Mapper" target="_blank">Food Database Mapper, hosted via HuggingFace Spaces</a> - With ZeroGPU for GPU-accelerated processing
- **Deployment**: Automatic CI/CD from GitHub to HuggingFace Spaces via GitHub Actions

## Overview

This tool helps researchers and nutritionists match food descriptions from various sources to standardized food databases. It's particularly useful for:

- Dietary assessment studies
- Food composition analysis  
- Nutritional database harmonization
- Menu item standardization
- Research data cleaning

### User-Friendly Interface

- **Step-by-step workflow**: Upload → Configure → Results
- **Sample dataset**: 25 pre-loaded examples for testing
- **Real-time processing**: Progress indicators and status updates
- **Flexible filtering**: Search and sort results dynamically
- **Adjustable threshold**: Control match sensitivity
- **CSV export**: Download results for further analysis

## Technical Specifications

### Performance
- Maximum dataset size: 50,000 items per file (demo limit)
- Batch processing for datasets over 30,000 items
- GPU-accelerated embeddings for faster processing

## How to Use

### Quick Start

1. **Upload Your Data**
   - Click "Load Sample Dataset" to try the tool
   - Or upload your own CSV files

2. **Select Columns**
   - Choose which columns contain food descriptions
   - Preview data to verify selection

3. **Configure Matching**
   - Select matching algorithm
   - Adjust similarity threshold (0.85 recommended)

4. **Review Results**
   - View matches with similarity scores
   - Filter for NO MATCH items
   - Download results as CSV

### Input Requirements

**Input CSV Format:**
```csv
id,food_description
1,"apple juice"
2,"chicken breast grilled"
3,"whole milk"
```

**Target/Reference CSV Format:**
```csv
code,food_name
A001,"Apple juice, unsweetened, bottled"
A002,"Chicken, broilers or fryers, breast, cooked"
A003,"Milk, whole, 3.25% milkfat"
```

## Understanding Results

### Similarity Scores

- **0.92-1.00**: Excellent match - Nearly identical items
- **0.88-0.92**: Very good match - Same food, different form
- **0.85-0.88**: Good match - Related foods
- **0.82-0.85**: Moderate match - Same category
- **Below 0.82**: Weak match - Consider as NO MATCH (if using default similarity threshold)

### NO MATCH Items

Items scoring below your threshold are marked as "NO MATCH". These require manual review or alternative matching strategies.

## Local Development

### Prerequisites

```bash
python >= 3.11
pip >= 23.0
```

### Installation

```bash
# Clone repository  
git clone https://github.com/RichardStoker-USDA/Food-Database-Mapper.git
cd Food-Database-Mapper

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

### Docker Deployment

```bash
# Build image
docker build -t food-mapping-gradio .

# Run container
docker run -p 7860:7860 food-mapping-gradio
```

## Important Limitations (Demo Version)

### Dataset Size Limits
- **Maximum 50,000 items per file** for the shared demo
- This limit ensures fair GPU usage across all users
- For larger datasets, please run locally with your own GPU

### GPU Resources
- Shared demo has a 2-minute GPU processing limit
- Daily GPU consumption caps apply
- Automatic batching for datasets over 30,000 items

### Local Installation for Large Datasets
For processing more than 50,000 items, install locally:
```bash
git clone https://github.com/RichardStoker-USDA/Food-Database-Mapper.git
cd Food-Database-Mapper
pip install -r requirements.txt
python app.py
```

## Data Privacy

- All processing happens in your session
- No data is stored permanently on servers
- Files are deleted after session ends
- No external API calls for data processing


## Support

For questions or issues:
- Email: richard.stoker@usda.gov
- GitHub Issues: [Report an issue](https://github.com/RichardStoker-USDA/Food-Database-Mapper/issues)
- HuggingFace Discussions: [Community discussions](https://huggingface.co/spaces/richtext/Food-Database-Mapper/discussions)

## License

CC0 1.0 Universal (CC0 1.0) - Public Domain Dedication

This work has been dedicated to the public domain under the Creative Commons CC0 1.0 Universal license. To the extent possible under law, USDA Agricultural Research Service has waived all copyright and related or neighboring rights to this work.

## Acknowledgments

Developed at the Western Human Nutrition Research Center, Davis, CA  
Diet, Microbiome and Immunity Research Unit  
United States Department of Agriculture | Agricultural Research Service

Based on research from the [USDA Food Description Mapping Project](https://github.com/mike-str/USDA-Food-Description-Mapping)
