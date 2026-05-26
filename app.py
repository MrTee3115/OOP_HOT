import gradio as gr
from evaluators import AIEvaluator, ColorTemperatureEvaluator

print("Loading models (this might take a few seconds)...")
# Initialize our polymorphic evaluators
ai_judge = AIEvaluator("hotness_model.pth")
color_judge = ColorTemperatureEvaluator()
print("Models loaded successfully! Launching UI...")


def analyze_image(img_path):
    if img_path is None:
        return "Please upload an image first.", ""

    # Process the image through both evaluators
    ai_result = ai_judge.evaluate(img_path)
    color_result = color_judge.evaluate(img_path)

    return ai_result, color_result


# Building the UI structure using Gradio Blocks (theme parameter removed from here)
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🤖 AI Face Rating & Color Temperature Evaluator
        Upload a portrait below to get a predicted beauty rating from our custom ResNet18 model, along with a pixel-level thermal color analysis.
        """
    )

    with gr.Row():
        # Left Column: Inputs
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Image Input (Drag & Drop)")

            gr.Examples(
                examples=["examples/low.jpg", "examples/medium.jpg", "examples/top.jpg"],
                inputs=image_input,
                label="Click on any example below to test:"
            )

            analyze_btn = gr.Button("Analyze Image", variant="primary")

        # Right Column: Outputs
        with gr.Column():
            ai_output = gr.Textbox(label="AI Rating (ResNet18)", lines=2)
            color_output = gr.Textbox(label="Color Heat Score", lines=2)

    # Connect the UI button to our python function
    analyze_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[ai_output, color_output]
    )

if __name__ == "__main__":
    # Launch the local server (theme parameter is moved here to fix the Gradio warning)
    demo.launch(theme=gr.themes.Soft())