"""
Food Database Mapper - Gradio Version
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
import spaces

# Import matching functions
from modules.matching_functions_batched import run_fuzzy_match, run_tfidf_match, run_embed_match, run_embed_match_batched
from modules.sample_data import get_sample_input_csv, get_sample_target_csv

# Professional CSS styling
custom_css = """
/* Main container */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* Headers and titles */
h1 {
    color: #2d5482 !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    margin-bottom: 0.5rem !important;
}

h2 {
    color: #3d5a80 !important;
    font-size: 1.8rem !important;
    margin-top: 2rem !important;
}

h3 {
    color: #475569 !important;
    font-size: 1.3rem !important;
    margin-top: 1.5rem !important;
}

/* Professional header */
.app-header {
    background: linear-gradient(135deg, #2d5482 0%, #4a7ba7 100%);
    color: white;
    padding: 2.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    text-align: center;
}

.app-header h1 {
    color: white !important;
    margin: 0 !important;
}

.app-header p {
    color: rgba(255, 255, 255, 0.9) !important;
    font-size: 1.1rem !important;
    margin-top: 0.5rem !important;
}

/* Tabs styling */
.tabs {
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 2rem;
}

button[role="tab"] {
    background: transparent !important;
    border: none !important;
    padding: 1rem 2rem !important;
    color: #64748b !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    border-bottom: 3px solid transparent !important;
    transition: all 0.3s !important;
}

button[role="tab"]:hover {
    color: #2d5482 !important;
    background: rgba(45, 84, 130, 0.05) !important;
}

button[role="tab"][aria-selected="true"] {
    color: #2d5482 !important;
    border-bottom: 3px solid #2d5482 !important;
    background: white !important;
}

/* Buttons */
.gr-button {
    font-weight: 500 !important;
    transition: all 0.3s !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
}

.gr-button.primary, button.primary {
    background: #2d5482 !important;
    color: white !important;
    border: none !important;
}

.gr-button.primary:hover, button.primary:hover {
    background: #1e3a5f !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(45, 84, 130, 0.3) !important;
}

.gr-button.secondary, button.secondary {
    background: white !important;
    color: #2d5482 !important;
    border: 2px solid #2d5482 !important;
}

.gr-button.secondary:hover, button.secondary:hover {
    background: #f0f4f8 !important;
}

/* Sample button special */
.sample-btn {
    background: #48bb78 !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
}

.sample-btn:hover {
    background: #38a169 !important;
}

/* File upload */
.file-upload {
    border: 2px dashed #cbd5e1 !important;
    border-radius: 8px !important;
    background: #f8fafc !important;
    padding: 2rem !important;
    transition: all 0.3s !important;
}

.file-upload:hover {
    border-color: #2d5482 !important;
    background: #f0f4f8 !important;
}

/* Dropdowns */
.gr-dropdown {
    border: 2px solid #e2e8f0 !important;
    border-radius: 8px !important;
}

.gr-dropdown:focus {
    border-color: #2d5482 !important;
    box-shadow: 0 0 0 3px rgba(45, 84, 130, 0.1) !important;
}

/* Info boxes */
.info-box {
    background: #f0f9ff !important;
    border: 1px solid #2d5482 !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    color: #1e3a5f !important;
    margin: 1rem 0 !important;
}

.success-box {
    background: #f0fdf4 !important;
    border: 1px solid #48bb78 !important;
    color: #166534 !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    margin: 1rem 0 !important;
}

.warning-box {
    background: #fffbeb !important;
    border: 1px solid #f59e0b !important;
    color: #92400e !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    margin: 1rem 0 !important;
}

/* Tables and Dataframes - Fix viewport expanding */
.gr-dataframe {
    border-radius: 8px !important;
    max-height: 400px !important;
    overflow: auto !important;
}

/* Constrain all dataframe containers */
div[class*="dataframe"] {
    max-height: 400px !important;
    overflow: auto !important;
}

/* Results table specific height */
#component-636 {
    max-height: 500px !important;
    overflow: auto !important;
}

/* Preview tables smaller height */
.gr-box .dataframe {
    max-height: 300px !important;
}

.gradio-container {
    overflow-x: hidden !important;
}

thead {
    background: #f1f5f9 !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 10 !important;
}

th {
    color: #475569 !important;
    font-weight: 600 !important;
    padding: 1rem !important;
}

td {
    padding: 0.75rem !important;
    border-bottom: 1px solid #e2e8f0 !important;
}

/* Footer */
.footer {
    margin-top: 4rem;
    padding: 2rem;
    border-top: 2px solid #e2e8f0;
    text-align: center;
    color: #64748b;
}
"""

def load_sample_data():
    """Load sample dataset for demonstration"""
    input_csv = get_sample_input_csv()
    target_csv = get_sample_target_csv()
    
    # Load DataFrames
    input_df = pd.read_csv(io.StringIO(input_csv))
    target_df = pd.read_csv(io.StringIO(target_csv))
    
    return (
        input_df,
        target_df,
        gr.update(choices=input_df.columns.tolist(), value=input_df.columns[1] if len(input_df.columns) > 1 else input_df.columns[0], visible=True),
        gr.update(choices=target_df.columns.tolist(), value=target_df.columns[1] if len(target_df.columns) > 1 else target_df.columns[0], visible=True),
        '<div class="success-box">✓ Sample data loaded successfully! Dataset includes 25 food items for testing.</div>',
        gr.update(visible=True, value=input_df.head(10)),
        gr.update(visible=True, value=target_df.head(10))
    )

def process_uploaded_file(file):
    """Process uploaded CSV file"""
    if file is None:
        return None, gr.update(choices=[], value=None), "", gr.update(visible=False)
    
    try:
        df = pd.read_csv(file.name)
        return (
            df,
            gr.update(choices=df.columns.tolist(), value=df.columns[0], visible=True),
            f'<div class="success-box">✓ Loaded {len(df)} rows, {len(df.columns)} columns</div>',
            gr.update(visible=True, value=df.head(10))
        )
    except Exception as e:
        return (
            None, 
            gr.update(choices=[], value=None),
            f'<div class="warning-box">Error reading file: {str(e)}</div>',
            gr.update(visible=False)
        )

def update_column_preview(df, column_name):
    """Update preview of selected column"""
    if df is None or column_name is None:
        return pd.DataFrame()
    
    try:
        sample_values = df[column_name].dropna().head(5).tolist()
        preview_df = pd.DataFrame({
            "Row": range(1, len(sample_values) + 1),
            "Sample Values": sample_values
        })
        return preview_df
    except:
        return pd.DataFrame()

@spaces.GPU(duration=120)  # 2 minutes GPU limit for shared demo
def run_matching_process(input_df, target_df, input_col, target_col, methods, threshold, clean_text, progress=gr.Progress()):
    """Execute the matching process with all algorithms"""
    
    if input_df is None or target_df is None:
        return None, '<div class="warning-box">Please upload both input and target CSV files</div>', ""
    
    if not input_col or not target_col:
        return None, '<div class="warning-box">Please select columns for matching</div>', ""
    
    if not methods:
        return None, '<div class="warning-box">Please select at least one matching method</div>', ""
    
    try:
        progress(0, desc="Preparing data...")
        
        # Prepare data lists
        input_list = input_df[input_col].dropna().tolist()
        target_list = target_df[target_col].dropna().tolist()
        
        # Check for dataset size limits (50k items max for shared demo)
        if len(input_list) > 50000:
            return None, f'''<div class="warning-box">
            <strong>Dataset Too Large</strong><br>
            Your input file contains {len(input_list):,} items, exceeding the 50,000 item limit.<br><br>
            This demo app is shared among multiple users and has daily GPU consumption limits.<br>
            For datasets larger than 50,000 items, please run this application locally using your own GPU resources.<br><br>
            Visit the <a href="https://github.com/mike-str/USDA-Food-Description-Mapping" target="_blank">GitHub repository</a> for local installation instructions.
            </div>''', ""
        
        if len(target_list) > 50000:
            return None, f'''<div class="warning-box">
            <strong>Dataset Too Large</strong><br>
            Your target file contains {len(target_list):,} items, exceeding the 50,000 item limit.<br><br>
            This demo app is shared among multiple users and has daily GPU consumption limits.<br>
            For datasets larger than 50,000 items, please run this application locally using your own GPU resources.<br><br>
            Visit the <a href="https://github.com/mike-str/USDA-Food-Description-Mapping" target="_blank">GitHub repository</a> for local installation instructions.
            </div>''', ""
        
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
        if "Semantic Embeddings" in methods:
            total_items = len(input_list) + len(target_list_unique)
            
            # Use batched processing for very large datasets
            if total_items > 20000:
                progress(current_progress, desc=f"Large dataset ({len(input_list)} inputs) - using batched processing...")
                
                # Define progress callback for batched processing
                def embedding_progress(p, desc):
                    # Scale progress within the method's allocated range
                    actual_progress = current_progress + (method_progress * p)
                    progress(actual_progress, desc=desc)
                
                embed_results = run_embed_match_batched(input_list, target_list_unique, progress_callback=embedding_progress)
            else:
                progress(current_progress + 0.05, desc="Loading embedding model...")
                progress(current_progress + 0.1, desc=f"Computing embeddings for {len(input_list)} inputs...")
                embed_results = run_embed_match(input_list, target_list_unique)
            
            results_df['best_match'] = embed_results['match']
            results_df['similarity_score'] = embed_results['score']
            results_df.loc[results_df['similarity_score'] < threshold, 'best_match'] = 'NO MATCH'
            
            current_progress += method_progress
            progress(current_progress, desc="Semantic matching complete")
        
        if "Fuzzy Matching" in methods:
            progress(current_progress + 0.05, desc="Running fuzzy matching...")
            
            fuzzy_results = run_fuzzy_match(input_list, target_list_unique, clean_text)
            results_df['fuzzy_match'] = fuzzy_results['match']
            results_df['fuzzy_score'] = [s/100.0 for s in fuzzy_results['score']]
            
            results_df.loc[results_df['fuzzy_score'] < threshold, 'fuzzy_match'] = 'NO MATCH'
            
            current_progress += method_progress
            progress(current_progress, desc="Fuzzy matching complete")
        
        if "TF-IDF" in methods:
            progress(current_progress + 0.05, desc="Running TF-IDF matching...")
            
            tfidf_results = run_tfidf_match(input_list, target_list_unique, clean_text)
            results_df['tfidf_match'] = tfidf_results['match']
            results_df['tfidf_score'] = tfidf_results['score']
            
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
            avg_score_str = f"{avg_score:.3f}" if not pd.isna(avg_score) else "N/A"
        else:
            no_matches = 0
            successful_matches = total_inputs
            avg_score_str = "N/A"
        
        status_msg = f'<div class="success-box">✓ Processing complete! Generated {len(results_df)} results.</div>'
        
        summary = f"""
        <div class="info-box">
            <strong>Results Summary:</strong><br>
            Total Inputs: {total_inputs}<br>
            Successful Matches: {successful_matches}<br>
            No Matches: {no_matches}<br>
            Average Score: {avg_score_str}
        </div>
        """
        
        progress(1.0, desc="Complete!")
        
        return results_df, status_msg, summary
        
    except Exception as e:
        return None, f'<div class="warning-box">Error: {str(e)}</div>', ""

def filter_results(results_df, search_term, show_no_match, sort_by_score):
    """Filter and sort results"""
    if results_df is None:
        return None
    
    filtered_df = results_df.copy()
    
    # Search filter
    if search_term and search_term.strip():
        search_lower = search_term.lower().strip()
        mask = filtered_df.apply(
            lambda row: row.astype(str).str.lower().str.contains(search_lower, na=False).any(), 
            axis=1
        )
        filtered_df = filtered_df[mask]
    
    # NO MATCH filter
    if show_no_match:
        if 'best_match' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['best_match'] == 'NO MATCH']
    
    # Sort by score
    if sort_by_score:
        score_cols = [col for col in filtered_df.columns if 'score' in col.lower()]
        if score_cols:
            filtered_df = filtered_df.sort_values(score_cols[0], ascending=False)
    
    return filtered_df

def export_results(results_df):
    """Export results to CSV"""
    if results_df is None:
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    results_df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return gr.update(visible=True, value=temp_file.name)

# Create Gradio interface
with gr.Blocks(css=custom_css, title="Food Database Mapper") as app:
    
    # State management
    input_df_state = gr.State(None)
    target_df_state = gr.State(None)
    results_df_state = gr.State(None)
    
    # Header
    gr.HTML("""
    <div class="app-header">
        <h1>Food Database Mapper</h1>
        <p>Western Human Nutrition Research Center | Davis, CA</p>
    </div>
    """)
    
    # Main interface with tabs
    with gr.Tabs(selected=0) as tabs:
        
        # Tab 1: Upload Files
        with gr.Tab("Step 1: Upload Files", id=0):
            gr.HTML('<div class="info-box">Upload your CSV files containing food descriptions to match, or try our sample dataset.</div>')
            
            # Sample data button
            with gr.Row():
                with gr.Column(scale=1):
                    pass
                with gr.Column(scale=2):
                    sample_btn = gr.Button(
                        "Load Sample Dataset (25 items)",
                        variant="secondary",
                        elem_classes=["sample-btn"]
                    )
                    sample_status = gr.HTML()
                with gr.Column(scale=1):
                    pass
            
            gr.HTML("<hr style='margin: 2rem 0; border: 1px solid #e2e8f0;'>")
            
            # File uploads
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Input File")
                    gr.Markdown("Upload CSV with descriptions to match")
                    input_file = gr.File(
                        label="Choose input CSV file",
                        file_types=[".csv"],
                        elem_classes=["file-upload"]
                    )
                    input_status = gr.HTML()
                    input_preview = gr.Dataframe(
                        label="Data Preview",
                        visible=False,
                        interactive=False,
                        wrap=True
                    )
                
                with gr.Column():
                    gr.Markdown("### Target File")
                    gr.Markdown("Upload CSV with reference descriptions")
                    target_file = gr.File(
                        label="Choose target CSV file",
                        file_types=[".csv"],
                        elem_classes=["file-upload"]
                    )
                    target_status = gr.HTML()
                    target_preview = gr.Dataframe(
                        label="Data Preview",
                        visible=False,
                        interactive=False,
                        wrap=True
                    )
            
            # Navigation button
            gr.HTML("<hr style='margin: 2rem 0; border: 1px solid #e2e8f0;'>")
            with gr.Row():
                with gr.Column(scale=1):
                    pass
                with gr.Column(scale=2):
                    continue_to_config_btn = gr.Button(
                        "Continue to Step 2: Configure →",
                        variant="primary",
                        size="lg"
                    )
                with gr.Column(scale=1):
                    pass
        
        # Tab 2: Configure
        with gr.Tab("Step 2: Configure", id=1):
            gr.HTML('<div class="info-box">Select columns and configure matching settings.</div>')
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Input Column")
                    input_column = gr.Dropdown(
                        label="Select column with descriptions:",
                        choices=[],
                        visible=False,
                        interactive=True
                    )
                    input_col_preview = gr.Dataframe(
                        label="Sample values",
                        interactive=False
                    )
                
                with gr.Column():
                    gr.Markdown("### Target Column")
                    target_column = gr.Dropdown(
                        label="Select column with references:",
                        choices=[],
                        visible=False,
                        interactive=True
                    )
                    target_col_preview = gr.Dataframe(
                        label="Sample values",
                        interactive=False
                    )
            
            gr.HTML("<hr style='margin: 2rem 0; border: 1px solid #e2e8f0;'>")
            
            gr.Markdown("### Matching Settings")
            
            with gr.Row():
                with gr.Column(scale=2):
                    methods = gr.CheckboxGroup(
                        label="Matching Methods:",
                        choices=["Semantic Embeddings", "Fuzzy Matching", "TF-IDF"],
                        value=["Semantic Embeddings"],
                        interactive=True
                    )
                
                with gr.Column(scale=1):
                    threshold = gr.Slider(
                        label="Similarity Threshold:",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.85,
                        step=0.05
                    )
                
                with gr.Column(scale=1):
                    clean_text = gr.Checkbox(
                        label="Apply text cleaning",
                        value=False
                    )
            
            gr.HTML("<hr style='margin: 2rem 0; border: 1px solid #e2e8f0;'>")
            with gr.Row():
                back_to_upload_btn = gr.Button("← Back to Step 1", variant="secondary")
                with gr.Column(scale=2):
                    pass
                run_btn = gr.Button("Run Matching Process", variant="primary", size="lg")
        
        # Tab 3: Results
        with gr.Tab("Step 3: Results", id=2):
            process_status = gr.HTML()
            process_summary = gr.HTML()
            
            gr.Markdown("### Filter Results")
            
            with gr.Row():
                search_box = gr.Textbox(
                    label="Search:",
                    placeholder="Type to filter...",
                    scale=2
                )
                show_no_match = gr.Checkbox(
                    label="Show only NO MATCH",
                    value=False,
                    scale=1
                )
                sort_by_score = gr.Checkbox(
                    label="Sort by score",
                    value=True,
                    scale=1
                )
            
            results_table = gr.Dataframe(
                label="Matching Results",
                interactive=False,
                wrap=True
            )
            
            gr.HTML("<hr style='margin: 2rem 0; border: 1px solid #e2e8f0;'>")
            with gr.Row():
                back_to_config_btn = gr.Button("← Back to Step 2", variant="secondary")
                download_file = gr.File(label="Download", visible=False)
                export_btn = gr.Button("Export to CSV", variant="primary")
                new_btn = gr.Button("Start New Matching", variant="secondary")
    
    # Footer
    gr.HTML("""
    <div class="footer">
        <strong>Western Human Nutrition Research Center</strong> | Davis, CA<br>
        Diet, Microbiome and Immunity Research Unit<br>
        United States Department of Agriculture | Agricultural Research Service
    </div>
    """)
    
    # Event handlers
    
    # Sample data
    sample_btn.click(
        fn=load_sample_data,
        outputs=[
            input_df_state, target_df_state,
            input_column, target_column,
            sample_status,
            input_preview, target_preview
        ]
    )
    
    # Navigation buttons
    continue_to_config_btn.click(
        fn=lambda: gr.update(selected=1),
        outputs=[tabs]
    )
    
    back_to_upload_btn.click(
        fn=lambda: gr.update(selected=0),
        outputs=[tabs]
    )
    
    back_to_config_btn.click(
        fn=lambda: gr.update(selected=1),
        outputs=[tabs]
    )
    
    # File uploads
    input_file.change(
        fn=process_uploaded_file,
        inputs=[input_file],
        outputs=[input_df_state, input_column, input_status, input_preview]
    )
    
    target_file.change(
        fn=process_uploaded_file,
        inputs=[target_file],
        outputs=[target_df_state, target_column, target_status, target_preview]
    )
    
    # Column previews
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
    
    # Run matching - navigate to Step 3 immediately, then run processing
    run_btn.click(
        fn=lambda: gr.update(selected=2),
        outputs=[tabs]
    ).then(
        fn=run_matching_process,
        inputs=[
            input_df_state, target_df_state,
            input_column, target_column,
            methods, threshold, clean_text
        ],
        outputs=[results_df_state, process_status, process_summary]
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
    
    # Export
    export_btn.click(
        fn=export_results,
        inputs=[results_df_state],
        outputs=[download_file]
    )
    
    # New analysis
    new_btn.click(
        fn=lambda: (
            None, None, None,
            gr.update(selected=0),
            gr.update(value=None),
            gr.update(value=None),
            "", "",
            gr.update(visible=False),
            gr.update(visible=False)
        ),
        outputs=[
            input_df_state, target_df_state, results_df_state,
            tabs,
            input_file, target_file,
            input_status, target_status,
            input_preview, target_preview
        ]
    )

# Launch
if __name__ == "__main__":
    app.launch()