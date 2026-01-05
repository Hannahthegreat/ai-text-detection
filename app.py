import gradio as gr
import joblib
import re
import numpy as np

# Load model
model_data = joblib.load('models/saved_models/final_model_lr.joblib')
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
    
# Create gradio interfae
with gr.Blocks(title="AI Text Detection", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# AI Text Detection System")
    gr.Markdown("### Detecting AI-generated text for academic integrity")
    gr.Markdown("**Model:** Logistic Regression | **Precision:** 84.2% | **Recall:** 80.0%")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(    
                label = "Enter text to analyze",
                placeholder = "Paste essay, paragraph, or any text here...",
                lines = 12
            )
            
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Analyze Text", variant="primary", scale=2)
        
        with gr.Column(scale=1):
            output_label = gr.Textbox(label="Prediction", interactive=False)
            output_confidence = gr.Textbox(label="Confidence", interactive=False)
            output_probs = gr.Textbox(label="Probability Distribution", interactive=False)
            output_stats = gr.Textbox(label="Text Statistics", interactive=False)

    # Examples
    gr.Markdown("### Tru these examples:")
    gr.Examples(
        examples=[
            ["The quick brown fox jumps over the lazy dog. This is a simple test sentence written by a human."],
                ["It is important to note that various factors contribute to this phenomenon. Additionally, one should consider the following aspects when analyzing the situation."],
                ["i cant believe this happened lol. basically what i mean is that the whole thing was kinda weird you know?"]
        ],
        inputs = input_text,
        label = "Click an example to test"
    )

    # Button actions
    submit_btn.click(
        fn=predict_text,
        inputs=input_text,
        outputs=[output_label, output_confidence, output_probs, output_stats]
    )

    clear_btn.click(
        fn = lambda: ("", "", "", ""),
        outputs=[output_label, output_confidence, output_probs, output_stats]
    )

    gr.Markdown("---")
    gr.Markdown("""
                **Student:** Hannah Soliao | **Module:** CPU6100-20 Machine Learning
                **Program:** BSc (Hons) Creative Computing @ Academic Centre RAK, UAE
                """)
if __name__ == "__main__":
    demo.launch()