import ast
import json
from contextlib import contextmanager

import gradio as gr
import nemo.collections.asr as nemo_asr
from nemo.collections.common.parts import MultiLayerPerceptron

# Workaround for a NeMo bug: GreedySequenceGenerator expects the classifier to be a
# TokenClassifier (which has a `.mlp`), but the SLURP model's classifier is a bare
# MultiLayerPerceptron. Making `.mlp` point back to the MLP itself satisfies the lookup.
if not hasattr(MultiLayerPerceptron, "mlp"):
    MultiLayerPerceptron.mlp = property(lambda self: self)


# The generator also uses TokenClassifier's with_log_softmax_enabled() context manager,
# which temporarily switches log-softmax on or off. Give the MLP the same behavior.
@contextmanager
def _with_log_softmax_enabled(self, value=True):
    previous = getattr(self, "log_softmax", True)
    self.log_softmax = value
    try:
        yield self
    finally:
        self.log_softmax = previous


if not hasattr(MultiLayerPerceptron, "with_log_softmax_enabled"):
    MultiLayerPerceptron.with_log_softmax_enabled = _with_log_softmax_enabled

MODEL_PATH = "/home/common/Downloads/nvidia-conformer-large-slurp-other-default-v1/slu_conformer_transformer_large_slurp_1.nemo"

print("Loading NeMo SLU model...")
# Use the concrete SLU class; ASRModel is abstract and can't be instantiated.
model = nemo_asr.models.SLUIntentSlotBPEModel.restore_from(restore_path=MODEL_PATH)
model.eval()


# Workaround for a second NeMo bug: transcribe() calls freeze()/unfreeze() on the
# model's submodules, but the SLU TransformerDecoder (and possibly others) is a plain
# torch module without those methods. Give any submodule that lacks them simple versions.
def _freeze(self, *args, **kwargs):
    # Remember which params were trainable so unfreeze(partial=True) can restore them
    self._frozen_by_patch = [p for p in self.parameters() if p.requires_grad]
    for p in self.parameters():
        p.requires_grad = False
    self.eval()


def _unfreeze(self, *args, partial=False, **kwargs):
    if partial:
        params = getattr(self, "_frozen_by_patch", [])
    else:
        params = list(self.parameters())
    for p in params:
        p.requires_grad = True
    self._frozen_by_patch = []


# Workaround for a third NeMo bug: the greedy generator now returns a tuple
# (tokens, samples, confidence, attention), but the SLU SequenceGenerator still
# expects only the tokens tensor. This wrapper hands back just the tokens.
class _TokensOnly:
    def __init__(self, generator):
        self._generator = generator

    def __call__(self, *args, **kwargs):
        out = self._generator(*args, **kwargs)
        if isinstance(out, tuple) and not kwargs.get("return_beam_scores", False):
            return out[0]
        return out

    def __getattr__(self, name):
        return getattr(self._generator, name)


if not isinstance(model.sequence_generator.generator, _TokensOnly):
    model.sequence_generator.generator = _TokensOnly(model.sequence_generator.generator)


for _name, _module in model.named_children():
    cls = type(_module)
    if not hasattr(cls, "freeze"):
        cls.freeze = _freeze
    if not hasattr(cls, "unfreeze"):
        cls.unfreeze = _unfreeze


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
    demo.launch(server_name="0.0.0.0", server_port=7860)
