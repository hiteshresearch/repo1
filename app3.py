import ast
import json
import os
import tempfile
from contextlib import contextmanager

import librosa
import numpy as np
import soundfile as sf

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

# Folder with one sub-folder per noise class, e.g.
#   noises/babble/cafe_01.wav
#   noises/traffic/street_02.wav
#   noises/white/white_01.wav
# The sub-folder name is shown as the noise class in the web interface.
NOISE_DIR = "/home/common/noises"
SNR_OPTIONS = [0, 5, 10]  # dB
TARGET_SR = 16000
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")

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


def scan_noise_dir(noise_dir):
    """Return {class_name: [(file_name, full_path), ...]} for every sub-folder."""
    catalog = {}
    if not os.path.isdir(noise_dir):
        print(f"WARNING: noise folder not found: {noise_dir}")
        return catalog
    for cls_name in sorted(os.listdir(noise_dir)):
        cls_path = os.path.join(noise_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
        files = [
            (f, os.path.join(cls_path, f))
            for f in sorted(os.listdir(cls_path))
            if f.lower().endswith(AUDIO_EXTS)
        ]
        if files:
            catalog[cls_name] = files
    return catalog


NOISE_CATALOG = scan_noise_dir(NOISE_DIR)
NOISE_CLASSES = list(NOISE_CATALOG.keys())
print(f"Found noise classes: {NOISE_CLASSES}")


def load_audio(path):
    """Load any audio file as 16 kHz mono float32."""
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio.astype(np.float32)


def add_noise_at_snr(speech, noise, snr_db, rng=None):
    """Mix noise into speech so the result has the requested SNR in dB."""
    rng = rng or np.random.default_rng()

    # Make the noise the same length as the speech: loop it if too short,
    # take a random segment if too long.
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    start = rng.integers(0, len(noise) - len(speech) + 1)
    noise = noise[start:start + len(speech)]

    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    if speech_power == 0 or noise_power == 0:
        return speech

    # SNR = 10 * log10(P_speech / P_noise_scaled)
    scale = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
    mixed = speech + scale * noise

    # Prevent clipping when saving as WAV (scales both equally, SNR is unchanged)
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


def save_temp_wav(audio):
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, audio, TARGET_SR)
    return path


def run_model(audio_path):
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


def predict_intent_and_slots(audio_path, use_noise, noise_path, snr_db):
    if audio_path is None:
        return None, "Please upload or record an audio file"
    try:
        speech = load_audio(audio_path)

        if use_noise:
            if not noise_path:
                return None, "Please select a noise sample, or untick 'Add background noise'"
            noise = load_audio(noise_path)
            speech = add_noise_at_snr(speech, noise, float(snr_db))

        model_input = save_temp_wav(speech)
        return model_input, run_model(model_input)
    except Exception as e:
        return None, f"An error occurred during inference: {e}"


def noise_choices(cls_name):
    return [(name, path) for name, path in NOISE_CATALOG.get(cls_name, [])]


def on_class_change(cls_name):
    choices = noise_choices(cls_name)
    first = choices[0][1] if choices else None
    return gr.update(choices=choices, value=first), first


def on_noise_toggle(use_noise):
    return gr.update(visible=use_noise)


with gr.Blocks(title="Conformer Large SLURP Demo") as demo:
    gr.Markdown("# Conformer Large SLURP Demo")
    gr.Markdown(
        "This web interface uses NVIDIA NeMo's Conformer Large model "
        "trained on the SLURP dataset for joint intent classification and slot filling. "
        "You can optionally mix background noise into your speech at a chosen SNR "
        "to test how robust the model is."
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Input Speech",
            )

            use_noise = gr.Checkbox(label="Add background noise", value=False)

            with gr.Group(visible=False) as noise_box:
                default_class = NOISE_CLASSES[0] if NOISE_CLASSES else None
                default_choices = noise_choices(default_class)
                default_noise = default_choices[0][1] if default_choices else None

                noise_class = gr.Dropdown(
                    choices=NOISE_CLASSES, value=default_class, label="Noise class"
                )
                noise_sample = gr.Dropdown(
                    choices=default_choices, value=default_noise, label="Noise sample"
                )
                noise_preview = gr.Audio(
                    value=default_noise, type="filepath",
                    label="Noise preview", interactive=False,
                )
                snr = gr.Radio(
                    choices=SNR_OPTIONS, value=SNR_OPTIONS[-1], label="SNR (dB)"
                )

            submit_btn = gr.Button("Analyse speech", variant="primary")

        with gr.Column():
            model_audio = gr.Audio(
                type="filepath",
                label="Audio sent to the model (16 kHz mono, with noise if selected)",
                interactive=False,
            )
            output_text = gr.Code(
                label="Extracted Semantics (Intent and Slots)",
                language="json",
            )

    use_noise.change(on_noise_toggle, inputs=use_noise, outputs=noise_box)
    noise_class.change(on_class_change, inputs=noise_class, outputs=[noise_sample, noise_preview])
    noise_sample.change(lambda p: p, inputs=noise_sample, outputs=noise_preview)

    submit_btn.click(
        fn=predict_intent_and_slots,
        inputs=[audio_input, use_noise, noise_sample, snr],
        outputs=[model_audio, output_text],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[NOISE_DIR],  # lets the page play the noise preview files
    )
