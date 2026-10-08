import ast
import json

import gradio as gr
import nemo.collections.asr as nemo_asr
from nemo.collections.common.parts import MultiLayerPerceptron

# Workaround for a NeMo bug: GreedySequenceGenerator expects the classifier to be a
# TokenClassifier (which has a `.mlp`), but the SLURP model's classifier is a bare
# MultiLayerPerceptron. Making `.mlp` point back to the MLP itself satisfies the lookup.
if not hasattr(MultiLayerPerceptron, "mlp"):
    MultiLayerPerceptron.mlp = property(lambda self: self)

MODEL_PATH = "/home/common/Downloads/nvidia-conformer-large-slurp-other-default-v1/slu_conformer_transformer_large_slurp_1.nemo"

print("Loading NeMo SLU model...")
# Use the concrete SLU class; ASRModel is abstract and can't be instantiated.
model = nemo_asr.models.SLUIntentSlotBPEModel.restore_from(restore_path=MODEL_PATH)
model.eval()


def predict_intent_and_slots(audio_path):
    if audio_path is None:
        return "Please upload or record an audio file"
    try:
        # SLU models expose transcribe(), not predict()
        predictions = model.transcribe([audio_path], batch_size=1)
        if not predictions:
            return "No predictions returned"
        raw = predictions[0]
        raw = getattr(raw, "text", raw)  # newer NeMo may return Hypothesis objects

        # SLURP output is usually a Python-style dict string (single quotes)
        for parser in (json.loads, ast.literal_eval):
            try:
                return json.dumps(parser(raw), indent=4)
            except Exception:
                pass
        return raw
    except Exception as e:
        return f"An error occurred during inference: {e}"


with gr.Blocks(title="Conformer Large SLURP Demo") as demo:
    gr.Markdown("# Conformer Large SLURP Demo")
    gr.Markdown(
        "This web interface uses NVIDIA NeMo's Conformer Large model "
        "trained on the SLURP dataset for joint intent classification and slot filling."
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Input Audio (16kHz WAV Mono)",
            )
            submit_btn = gr.Button("Analyse speech", variant="primary")

        with gr.Column():
            output_text = gr.Code(
                label="Extracted Semantics (Intent and Slots)",
                language="json",
            )

    submit_btn.click(fn=predict_intent_and_slots, inputs=audio_input, outputs=output_text)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6000)
