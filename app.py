#!/usr/bin/env python3
"""
Gradio web app for STEM Animation Generator
Simple UI: Input concept -> Progress bar -> Video player
"""

import gradio as gr
from pipeline import Pipeline
from pathlib import Path

pipeline = Pipeline()

def generate_animation(concept: str, language: str = "English", progress=gr.Progress()):
    """
    Main generation function called by Gradio

    Args:
        concept: User input STEM concept
        language: Target language for narration (English, Chinese, Spanish, Vietnamese)
        progress: Gradio progress tracker

    Returns:
        Video file path or error message
    """
    if not concept or concept.strip() == "":
        return None
    
    def update_progress(message: str, percentage: float):
        progress(percentage, desc=message)
    
    result = pipeline.run(concept, progress_callback=update_progress, target_language=language)
    
    if result["status"] == "success" and result.get("video_result"):
        video_path = result["video_result"]["output_path"]
        if Path(video_path).exists():
            return video_path
        else:
            return None
    else:
        return None

with gr.Blocks(title="STEMViz") as demo:
    gr.Markdown("# STEMViz")
    gr.Markdown("Transform STEM concepts into narrated educational animations")
    
    with gr.Row():
        with gr.Column():
            concept_input = gr.Textbox(
                label="Enter STEM Concept",
                placeholder="e.g., Explain Bubble Sort, Bayes' Theorem, Gradient Descent...",
                lines=2
            )
            language_dropdown = gr.Dropdown(
                choices=["English", "Chinese", "Spanish", "Vietnamese"],
                value="English",
                label="Narration Language"
            )
            generate_btn = gr.Button("Generate Animation", variant="primary")
        
    with gr.Row():
        video_output = gr.Video(
            label="Generated Animation",
            autoplay=True
        )
    
    gr.Examples(
        examples=[
            ["Explain Bubble Sort"],
            ["Explain Bayes' Theorem"],
            ["Explain Gradient Descent"]
        ],
        inputs=concept_input
    )
    
    generate_btn.click(
        fn=generate_animation,
        inputs=[concept_input, language_dropdown],
        outputs=video_output
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
