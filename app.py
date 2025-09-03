"""
Food Description Semantic Mapping Tool - Gradio Version
USDA Agricultural Research Service
Western Human Nutrition Research Center
"""

import gradio as gr
import pandas as pd
import numpy as np
import io
import time
from datetime import datetime
from pathlib import Path
import tempfile
import os

# Import matching functions
from modules.matching_functions import run_fuzzy_match, run_tfidf_match, run_embed_match
from modules.sample_data import get_sample_input_csv, get_sample_target_csv

# Custom CSS matching original Streamlit styling
custom_css = """
/* Main container and layout */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
}

/* Header styling */
.header-container {
    background: linear-gradient(135deg, #2d5482 0%, #5a7da5 100%);
    color: white !important;
    padding: 2rem;
    margin: -1rem -1rem 2rem -1rem;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
}

.main-title {
    font-size: 2rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
    color: white !important;
    letter-spacing: -0.5px;
}

.subtitle {
    margin: 0.5rem 0 0 0 !important;
    opacity: 0.9;
    font-size: 1rem !important;
    color: white !important;
}

/* Tab styling */
.tab-nav button {
    background-color: #f7f9fb !important;
    color: #3d5a80 !important;
    border: none !important;
    padding: 1rem 1.5rem !important;
    font-weight: 500 !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.3s !important;
}

.tab-nav button:hover {
    background-color: #e8f1f9 !important;
    color: #2d5482 !important;
}

.tab-nav button.selected {
    background-color: #2d5482 !important;
    color: white !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.12) !important;
}

/* Button styling to match Streamlit */
.gradio-button {
    background-color: #5a7da5 !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 5px !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.3s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}

.gradio-button:hover {
    background-color: #4a6d8f !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 6px rgba(0,0,0,0.12) !important;
}

/* Sample data button special styling */
.sample-button {
    background-color: #48bb78 !important;
    border: 2px solid #38a169 !important;
}

.sample-button:hover {
    background-color: #38a169 !important;
}

/* File upload area */
.file-upload {
    border: 2px dashed #5a7da5 !important;
    border-radius: 8px !important;
    padding: 2rem !important;
    background-color: #f7f9fb !important;
    transition: all 0.3s !important;
}

.file-upload:hover {
    border-color: #2d5482 !important;
    background-color: #e8f1f9 !important;
}

/* Cards and containers */
.gradio-container .gradio-column {
    background: white !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
    margin: 0.5rem !important;
    padding: 1rem !important;
}

/* Info boxes */
.info-box {
    background-color: #f5f9fc !important;
    color: #3d5a80 !important;
    border: 1px solid #b8d4e8 !important;
    border-radius: 5px !important;
    padding: 1rem !important;
    margin: 1rem 0 !important;
    border-left: 4px solid #2d5482 !important;
}

/* Success messages */
.success-box {
    background-color: #f0fdf4 !important;
    color: #166534 !important;
    border: 1px solid #48bb78 !important;
    border-radius: 5px !important;
    padding: 1rem !important;
    margin: 1rem 0 !important;
    border-left: 4px solid #48bb78 !important;
}

/* Progress bar */
.progress-container {
    background-color: #f7f9fb !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    margin: 1rem 0 !important;
    border-left: 4px solid #2d5482 !important;
}

/* Dataframe styling */
.dataframe {
    border-radius: 8px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08) !important;
}

/* Input styling */
.gradio-textbox input, .gradio-dropdown select, .gradio-slider input {
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    transition: border-color 0.3s !important;
}

.gradio-textbox input:focus, .gradio-dropdown select:focus {
    border-color: #2d5482 !important;
    box-shadow: 0 0 0 3px rgba(45, 84, 130, 0.1) !important;
}

/* Footer styling */
.footer {
    text-align: center;
    color: #5a6c7d !important;
    font-size: 0.9rem !important;
    padding: 2rem 1rem !important;
    margin-top: 3rem !important;
    border-top: 2px solid #e2e8f0 !important;
    background: white !important;
    border-radius: 8px !important;
}

.footer a {
    color: #2d5482 !important;
    text-decoration: none !important;
}

.footer a:hover {
    color: #5a7da5 !important;
    text-decoration: underline !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .main-title {
        font-size: 1.5rem !important;
    }
    
    .header-container {
        padding: 1.5rem 1rem !important;
    }
    
    .tab-nav button {
        padding: 0.75rem 1rem !important;
        font-size: 0.9rem !important;
    }
}
"""

def load_sample_data():
    """Load sample dataset for demonstration"""
    input_csv = get_sample_input_csv()
    target_csv = get_sample_target_csv()
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(input_csv)
        input_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(target_csv)
        target_file = f.name
    
    # Load DataFrames
    input_df = pd.read_csv(input_file)
    target_df = pd.read_csv(target_file)
    
    # Clean up temp files
    os.unlink(input_file)
    os.unlink(target_file)
    
    return (
        input_file,  # Return path for file component
        target_file,  # Return path for file component  
        input_df,    # Return DataFrame for preview
        target_df,   # Return DataFrame for preview
        input_df.columns.tolist(),  # Column choices for input
        target_df.columns.tolist(), # Column choices for target
        input_df.columns[1] if len(input_df.columns) > 1 else input_df.columns[0],  # Default input column
        target_df.columns[1] if len(target_df.columns) > 1 else target_df.columns[0], # Default target column
        f"✅ Sample data loaded successfully! Dataset includes 25 food items with varying similarity scores for testing.",
        gr.update(visible=True),  # Show input preview
        gr.update(visible=True)   # Show target preview
    )

def process_uploaded_file(file):
    """Process uploaded CSV file"""
    if file is None:
        return None, [], None, "Please upload a CSV file", gr.update(visible=False)
    
    try:
        df = pd.read_csv(file.name)
        preview_html = df.head(10).to_html(classes="dataframe", index=False)
        return (
            df, 
            df.columns.tolist(), 
            df.columns[0] if len(df.columns) > 0 else None,
            f"✅ Loaded {len(df)} rows from CSV file",
            gr.update(visible=True, value=preview_html)
        )
    except Exception as e:
        return None, [], None, f"❌ Error reading CSV file: {str(e)}", gr.update(visible=False)

def update_column_preview(df, column_name):
    """Update preview of selected column"""
    if df is None or column_name is None or column_name not in df.columns:
        return "Select a column to see sample values"
    
    sample_values = df[column_name].dropna().head(5)
    preview_text = "Sample values from selected column:\n"
    for i, val in enumerate(sample_values, 1):
        val_str = str(val)
        if len(val_str) > 80:
            val_str = val_str[:80] + "..."
        preview_text += f"{i}. {val_str}\n"
    
    return preview_text

def run_matching_process(input_df, target_df, input_col, target_col, methods, threshold, clean_text, progress=gr.Progress()):
    """Execute the matching process with all algorithms"""
    
    if input_df is None or target_df is None:
        return None, "❌ Please upload both input and target CSV files", ""
    
    if not input_col or not target_col:
        return None, "❌ Please select columns for matching", ""
    
    if not methods:
        return None, "❌ Please select at least one matching method", ""
    
    if input_col not in input_df.columns:
        return None, f"❌ Column '{input_col}' not found in input file", ""
    
    if target_col not in target_df.columns:
        return None, f"❌ Column '{target_col}' not found in target file", ""
    
    try:
        progress(0, desc="Preparing data...")
        
        # Prepare data lists
        input_list = input_df[input_col].dropna().tolist()
        target_list = target_df[target_col].dropna().tolist()
        
        # Remove duplicates from target list
        target_list_unique = list(dict.fromkeys(target_list))
        
        progress(0.1, desc=f"Processing {len(input_list)} inputs against {len(target_list_unique)} unique targets...")
        
        # Initialize results
        results_df = pd.DataFrame({
            'input_description': input_list
        })
        
        total_methods = len(methods)
        method_progress = 0.8 / total_methods
        current_progress = 0.1
        
        # Run each selected method
        if "embed" in methods:
            progress(current_progress + 0.05, desc="Loading semantic embedding model...")
            progress(current_progress + 0.1, desc="Computing semantic embeddings (this may take a moment)...")
            
            embed_results = run_embed_match(input_list, target_list_unique)
            
            # Apply threshold
            results_df['best_match'] = embed_results['match']
            results_df['similarity_score'] = embed_results['score']
            results_df.loc[results_df['similarity_score'] < threshold, 'best_match'] = 'NO MATCH'
            
            current_progress += method_progress
            progress(current_progress, desc="Semantic matching complete")
        
        if "fuzzy" in methods:
            progress(current_progress + 0.05, desc="Running fuzzy string matching...")
            
            fuzzy_results = run_fuzzy_match(input_list, target_list_unique, clean_text)
            results_df['fuzzy_match'] = fuzzy_results['match']
            results_df['fuzzy_score'] = [s/100.0 for s in fuzzy_results['score']]  # Normalize to 0-1
            
            # Apply threshold
            results_df.loc[results_df['fuzzy_score'] < threshold, 'fuzzy_match'] = 'NO MATCH'
            
            current_progress += method_progress
            progress(current_progress, desc="Fuzzy matching complete")
        
        if "tfidf" in methods:
            progress(current_progress + 0.05, desc="Running TF-IDF matching...")
            
            tfidf_results = run_tfidf_match(input_list, target_list_unique, clean_text)
            results_df['tfidf_match'] = tfidf_results['match']
            results_df['tfidf_score'] = tfidf_results['score']
            
            # Apply threshold
            results_df.loc[results_df['tfidf_score'] < threshold, 'tfidf_match'] = 'NO MATCH'
            
            current_progress += method_progress
            progress(current_progress, desc="TF-IDF matching complete")
        
        # Round scores for display
        for col in results_df.columns:
            if 'score' in col:
                results_df[col] = results_df[col].round(4)
        
        progress(0.95, desc="Finalizing results...")
        
        # Generate summary statistics
        total_inputs = len(results_df)
        if 'best_match' in results_df.columns:
            no_matches = (results_df['best_match'] == 'NO MATCH').sum()
            successful_matches = total_inputs - no_matches
            avg_score = results_df[results_df['best_match'] != 'NO MATCH']['similarity_score'].mean()
            avg_score = f"{avg_score:.3f}" if not pd.isna(avg_score) else "N/A"
        else:
            no_matches = 0
            successful_matches = total_inputs
            avg_score = "N/A"
        
        summary = f"""
        📊 **Processing Results:**
        - **Total Inputs:** {total_inputs:,}
        - **Successful Matches:** {successful_matches:,}
        - **No Matches:** {no_matches:,}
        - **Average Match Score:** {avg_score}
        """
        
        progress(1.0, desc="Processing complete!")
        
        return results_df, f"✅ Processing complete! Generated {len(results_df):,} results.", summary
        
    except Exception as e:
        return None, f"❌ Error during processing: {str(e)}", ""

def filter_results(results_df, search_term, show_no_match, sort_by_score):
    """Filter and sort results based on user preferences"""
    if results_df is None:
        return None
    
    filtered_df = results_df.copy()
    
    # Apply search filter
    if search_term and search_term.strip():
        search_lower = search_term.lower().strip()
        mask = filtered_df.apply(
            lambda row: row.astype(str).str.lower().str.contains(search_lower, na=False).any(), 
            axis=1
        )
        filtered_df = filtered_df[mask]
    
    # Apply NO MATCH filter
    if show_no_match and 'best_match' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['best_match'] == 'NO MATCH']
    
    # Apply sorting
    if sort_by_score and 'similarity_score' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('similarity_score', ascending=False)
    
    return filtered_df

def download_results(results_df):
    """Prepare results for download"""
    if results_df is None:
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"semantic_matching_results_{timestamp}.csv"
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        results_df.to_csv(f.name, index=False)
        return f.name

# Create Gradio interface
with gr.Blocks(css=custom_css, title="Food Description Semantic Mapping Tool - USDA ARS") as app:
    
    # Header
    gr.HTML("""
    <div class="header-container">
        <h1 class="main-title">Food Description Semantic Mapping Tool</h1>
        <p class="subtitle">United States Department of Agriculture - Agricultural Research Service</p>
    </div>
    """)
    
    # State variables
    input_df_state = gr.State(None)
    target_df_state = gr.State(None)
    results_df_state = gr.State(None)
    
    with gr.Tabs() as tabs:
        # Step 1: Upload Files
        with gr.Tab("Step 1: Upload Files", id="upload") as tab1:
            
            gr.HTML('<div class="info-box">Upload your CSV files containing food descriptions to match, or try the sample dataset to see how the tool works.</div>')
            
            # Sample data section
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML("")  # Spacer
                with gr.Column(scale=3):
                    sample_btn = gr.Button(
                        "🧪 Load Sample Dataset (25 items)", 
                        variant="secondary",
                        elem_classes=["sample-button"]
                    )
                    sample_status = gr.HTML()
                with gr.Column(scale=2):
                    gr.HTML("")  # Spacer
            
            gr.HTML('<hr style="margin: 2rem 0;">')
            gr.HTML('<h3 style="color: #3d5a80;">Or Upload Your Own Files</h3>')
            
            with gr.Row(equal_height=True):
                # Input file upload
                with gr.Column(scale=1):
                    gr.HTML('<h4 style="color: #2d5482;">Input File</h4>')
                    gr.HTML('<p>Upload a CSV file containing the descriptions you want to match</p>')
                    
                    input_file = gr.File(
                        label="Choose your input CSV file",
                        file_types=[".csv"],
                        elem_classes=["file-upload"]
                    )
                    input_status = gr.HTML()
                    input_preview = gr.HTML(visible=False)
                
                # Target file upload  
                with gr.Column(scale=1):
                    gr.HTML('<h4 style="color: #2d5482;">Target File</h4>')
                    gr.HTML('<p>Upload a CSV file containing the reference descriptions to match against</p>')
                    
                    target_file = gr.File(
                        label="Choose your target CSV file", 
                        file_types=[".csv"],
                        elem_classes=["file-upload"]
                    )
                    target_status = gr.HTML()
                    target_preview = gr.HTML(visible=False)
            
            # Navigation
            with gr.Row():
                gr.HTML("")  # Spacer
                continue_btn = gr.Button(
                    "Continue to Column Selection →", 
                    variant="primary",
                    size="lg"
                )
        
        # Step 2: Column Selection and Configuration
        with gr.Tab("Step 2: Select Columns", id="configure") as tab2:
            
            gr.HTML('<div class="info-box">Select which columns contain the food descriptions for matching and configure your matching settings.</div>')
            
            with gr.Row(equal_height=True):
                # Input column configuration
                with gr.Column(scale=1):
                    gr.HTML('<h4 style="color: #2d5482;">Input Data Configuration</h4>')
                    input_info = gr.HTML()
                    input_column = gr.Dropdown(
                        label="Select column containing descriptions to match:",
                        choices=[],
                        interactive=True
                    )
                    input_col_preview = gr.Textbox(
                        label="Sample values from selected column:",
                        lines=5,
                        interactive=False
                    )
                
                # Target column configuration
                with gr.Column(scale=1):
                    gr.HTML('<h4 style="color: #2d5482;">Target Data Configuration</h4>')
                    target_info = gr.HTML()
                    target_column = gr.Dropdown(
                        label="Select column containing reference descriptions:",
                        choices=[],
                        interactive=True
                    )
                    target_col_preview = gr.Textbox(
                        label="Sample values from selected column:",
                        lines=5,
                        interactive=False
                    )
            
            # Advanced settings
            gr.HTML('<h4 style="color: #2d5482; margin-top: 2rem;">Advanced Settings</h4>')
            
            with gr.Row():
                methods = gr.CheckboxGroup(
                    label="Matching Methods:",
                    choices=[
                        ("embed", "Semantic Embeddings (Recommended)"),
                        ("fuzzy", "Fuzzy String Matching"), 
                        ("tfidf", "TF-IDF Similarity")
                    ],
                    value=["embed"]
                )
                
                with gr.Column():
                    threshold = gr.Slider(
                        label="Similarity Threshold for NO MATCH:",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.85,
                        step=0.05,
                        info="Matches below this threshold will be marked as NO MATCH. For embedding models, try 0.85 or higher."
                    )
                    
                    clean_text = gr.Checkbox(
                        label="Apply text cleaning",
                        value=False,
                        info="Note: Text cleaning is NOT recommended for embedding models as they work better with original text"
                    )
            
            # Navigation
            with gr.Row():
                back_step1_btn = gr.Button("← Back to File Upload")
                gr.HTML("")  # Spacer
                run_matching_btn = gr.Button(
                    "Run Matching Process",
                    variant="primary", 
                    size="lg"
                )
        
        # Step 3: Results  
        with gr.Tab("Step 3: View Results", id="results") as tab3:
            
            gr.HTML('<div class="info-box">Review your matching results, filter as needed, and download the complete dataset.</div>')
            
            # Processing status
            process_status = gr.HTML()
            process_summary = gr.HTML()
            
            # Results filtering
            gr.HTML('<h4 style="color: #2d5482;">Filter and Search Results</h4>')
            with gr.Row():
                search_box = gr.Textbox(
                    label="Search/filter results:",
                    placeholder="Type to filter...",
                    scale=2
                )
                show_no_match = gr.Checkbox(
                    label="Show only NO MATCH items",
                    value=False,
                    scale=1
                )
                sort_by_score = gr.Checkbox(
                    label="Sort by similarity score", 
                    value=True,
                    scale=1
                )
            
            # Results table
            results_table = gr.Dataframe(
                label="Detailed Results",
                interactive=False,
                wrap=True,
                elem_classes=["dataframe"]
            )
            
            # Download and navigation
            with gr.Row():
                back_step2_btn = gr.Button("← Back to Column Selection")
                download_btn = gr.Button(
                    "📥 Download Results (CSV)",
                    variant="primary"
                )
                new_analysis_btn = gr.Button("Start New Analysis →")
    
    # Footer
    gr.HTML("""
    <div class="footer">
        <strong>Western Human Nutrition Research Center</strong> | Davis, CA<br>
        Diet, Microbiome and Immunity Research Unit<br>
        United States Department of Agriculture | Agricultural Research Service<br>
        <br>
        <small>Based on <a href='https://github.com/mike-str/USDA-Food-Description-Mapping' target='_blank'>USDA Food Description Mapping Research</a></small>
    </div>
    """)
    
    # Event handlers
    
    # Sample data loading
    sample_btn.click(
        fn=load_sample_data,
        outputs=[
            input_file, target_file, input_df_state, target_df_state,
            input_column, target_column, input_column, target_column,
            sample_status, input_preview, target_preview
        ]
    ).then(
        fn=lambda df: f"File loaded: {len(df)} rows" if df is not None else "",
        inputs=[input_df_state],
        outputs=[input_info]
    ).then(
        fn=lambda df: f"File loaded: {len(df)} rows" if df is not None else "",
        inputs=[target_df_state], 
        outputs=[target_info]
    )
    
    # File upload handlers
    input_file.change(
        fn=process_uploaded_file,
        inputs=[input_file],
        outputs=[input_df_state, input_column, input_column, input_status, input_preview]
    ).then(
        fn=lambda df: f"File loaded: {len(df)} rows" if df is not None else "",
        inputs=[input_df_state],
        outputs=[input_info]
    )
    
    target_file.change(
        fn=process_uploaded_file,
        inputs=[target_file],
        outputs=[target_df_state, target_column, target_column, target_status, target_preview]
    ).then(
        fn=lambda df: f"File loaded: {len(df)} rows" if df is not None else "",
        inputs=[target_df_state],
        outputs=[target_info]
    )
    
    # Column selection handlers
    input_column.change(
        fn=update_column_preview,
        inputs=[input_df_state, input_column],
        outputs=[input_col_preview]
    )
    
    target_column.change(
        fn=update_column_preview,
        inputs=[target_df_state, target_column],
        outputs=[target_col_preview]
    )
    
    # Navigation handlers
    continue_btn.click(
        fn=lambda: gr.update(selected="configure"),
        outputs=[tabs]
    )
    
    back_step1_btn.click(
        fn=lambda: gr.update(selected="upload"),
        outputs=[tabs]
    )
    
    back_step2_btn.click(
        fn=lambda: gr.update(selected="configure"), 
        outputs=[tabs]
    )
    
    # Matching process
    run_matching_btn.click(
        fn=run_matching_process,
        inputs=[
            input_df_state, target_df_state, input_column, target_column,
            methods, threshold, clean_text
        ],
        outputs=[results_df_state, process_status, process_summary]
    ).then(
        fn=lambda: gr.update(selected="results"),
        outputs=[tabs]
    ).then(
        fn=filter_results,
        inputs=[results_df_state, search_box, show_no_match, sort_by_score],
        outputs=[results_table]
    )
    
    # Results filtering
    search_box.change(
        fn=filter_results,
        inputs=[results_df_state, search_box, show_no_match, sort_by_score],
        outputs=[results_table]
    )
    
    show_no_match.change(
        fn=filter_results,
        inputs=[results_df_state, search_box, show_no_match, sort_by_score],
        outputs=[results_table]
    )
    
    sort_by_score.change(
        fn=filter_results,
        inputs=[results_df_state, search_box, show_no_match, sort_by_score],
        outputs=[results_table]
    )
    
    # Download handler
    download_btn.click(
        fn=download_results,
        inputs=[results_df_state],
        outputs=[gr.File()]
    )
    
    # New analysis handler
    new_analysis_btn.click(
        fn=lambda: [None, None, None, gr.update(selected="upload"), "", ""],
        outputs=[input_df_state, target_df_state, results_df_state, tabs, process_status, process_summary]
    )

# Launch the app
if __name__ == "__main__":
    app.launch()