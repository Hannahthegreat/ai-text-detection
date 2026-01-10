import gradio as gr
import joblib
import re
import numpy as np

# Load model
model_data = joblib.load('models/saved_models/final_ensemble_lr_xgb.joblib')
model = model_data['model']
vectorizer = model_data['vectorizer']

def clean_text(text):
    # Clean text same way as training data
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def predict_text(text):
    # Predict if text is AI generated or human-written
    if not text.strip():
        return "Please enter text to analyze", "", "",""
    
    # Preprocess
    cleaned = clean_text(text)
    
    # Vectorize
    text_vector = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    
    # Format results
    label = "AI Generated" if prediction == 1 else "Human-Written"
    confidence = probabilities[prediction] * 100
    
    ai_prob = probabilities[1] * 100
    human_prob = probabilities[0] * 100
    
    # Text statistics
    word_count = len(cleaned.split())
    char_count = len(cleaned)
    
    return (
        label,
        f"{confidence:.1f}%",
        f"AI: {ai_prob:.1f}% | Human: {human_prob:.1f}%",
        f"Words: {word_count} | Characters: {char_count}"
    )

# Load CSS from external file
with open("style.css", "r") as f:
    custom_css = f.read()

with gr.Blocks(title="AI Text Detection", theme=gr.themes.Soft(), css=custom_css) as demo:
    
    # Header with HTML
    gr.HTML("""
        <div class="header-section">
            <h1 class="header-title">AI Text Detection System</h1>
            <p class="header-subtitle">Detecting AI-generated text for academic integrity</p>
            <div class="header-stats">
                <strong>Model:</strong> LR + XGBoost Ensemble | 
                <strong>Precision:</strong> 85.16% | 
                <strong>Recall:</strong> 81.24% | 
                <strong>Accuracy:</strong> 91.39%
            </div>
        </div>
    """)
    
    # Info card
    gr.HTML("""
        <div class="info-card">
            <p style="margin: 0;">
                <strong>How it works:</strong> This system uses an ensemble of machine learning models 
                trained on linguistic features to distinguish between human-written and AI-generated text. 
                Enter your text below and click "Analyze Text" to get started.
            </p>
        </div>
    """)
    
    # Main interface
    with gr.Row():
        with gr.Column(scale=1, elem_classes="input-column"):
            gr.Markdown("### Input Text")
            input_text = gr.Textbox(    
                label="Enter text to analyze",
                placeholder="Paste essay, paragraph, or any text here...\n\nMinimum 50 characters recommended for accurate prediction.",
                lines=12,
                elem_classes="textbox-input"
            )
            
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary", elem_classes="secondary-button")
                submit_btn = gr.Button("Analyze Text", variant="primary", scale=2, elem_classes="primary-button")
        
        with gr.Column(scale=1, elem_classes="output-column"):
            gr.Markdown("### Analysis Results")
            output_label = gr.Textbox(label="Prediction", interactive=False, elem_classes="output-prediction")
            output_confidence = gr.Textbox(label="Confidence Level", interactive=False, elem_classes="output-confidence")
            output_probs = gr.Textbox(label="Probability Distribution", interactive=False, elem_classes="output-probs")
            output_stats = gr.Textbox(label="Text Statistics", interactive=False, elem_classes="output-stats")
            
    # Examples section
    with gr.Row(elem_classes="examples-section"):
        with gr.Column():
            gr.Markdown("### Try These Examples")
            gr.Markdown("Click any example below to test the detector with different writing styles:")
            
            gr.Examples(
                examples=[
                    ["The quick brown fox jumps over the lazy dog. This is a simple test sentence written by a human. I personally think that writing authentically requires emotional depth and personal experience that machines can't replicate."],
                    ["It is important to note that various factors contribute to this phenomenon. Additionally, one should consider the following aspects when analyzing the situation. Furthermore, the implications of these findings are significant."],
                    ["i cant believe this happened lol. basically what i mean is that the whole thing was kinda weird you know? like it doesn't make sense at all tbh"]
                ],
                inputs=input_text,
                label="Sample Texts"
            )

    # Usage tips
    gr.HTML("""
        <div class="info-card" style="margin-top: 1rem;">
            <p style="margin: 0; color: #555;">
                <strong>Tips for best results:</strong>
                <ul style="margin: 0.5rem 0 0 1.5rem;">
                    <li>Use at least 50 characters for reliable predictions</li>
                    <li>The model works best on complete sentences and paragraphs</li>
                    <li>Context matters - academic writing patterns differ from casual text</li>
                    <li>Remember: No AI detector is 100% accurate - use as one tool among many</li>
                </ul>
            </p>
        </div>
    """)

    # Button actions
    submit_btn.click(
        fn=predict_text,
        inputs=input_text,
        outputs=[output_label, output_confidence, output_probs, output_stats]
    )

    clear_btn.click(
        fn=lambda: ("", "", "", ""),
        outputs=[output_label, output_confidence, output_probs, output_stats]
    )

    # Footer
    gr.HTML("""
        <div class="footer-section">
            <p style="margin: 0 0 0.5rem 0; font-size: 0.95rem;">
                <strong>Student:</strong> Hannah Soliao | 
                <strong>Module:</strong> CPU6100-20 Machine Learning
            </p>
            <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">
                <strong>Program:</strong> BSc (Hons) Creative Computing @ Academic Centre RAK, UAE
            </p>
            <p style="margin: 1rem 0 0 0; font-size: 0.85rem; opacity: 0.7;">
                Built with Python, scikit-learn, and Gradio | January 2025
            </p>
        </div>
    """)

if __name__ == "__main__":
    demo.launch()