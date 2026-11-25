"""
PS5 - Llama 3.2 1B operator advice, running fully offline on the Raspberry Pi with ExecuTorch.

Needs (from Google Drive -> Textile fabric -> llama):
    llama/model.pte        (Llama 3.2 1B Instruct, exported with optimum-cli, 8da4w)
    llama/tokenizer.json

Test on the Pi:
    source ~/ps5env/bin/activate
    python llama_advice.py --defect hole_burst --severity High --size 6.1
"""
import argparse, time

# These two imports register the extra kernels the exported Llama needs
# (8-bit embedding + fast attention / KV-cache). Keep them before creating the runner.
import executorch.kernels.quantized  # noqa: F401
try:
    import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401
except Exception as e:  # still works without it on some builds
    print("note: custom ops not loaded:", e)
from executorch.extension.llm.runner import TextLLMRunner, GenerationConfig

SYSTEM = ("You are a quality-control assistant in a textile factory. "
          "Answer in at most 50 words, each action under 10 words, plain text, using exactly this format:\n"
          "Likely cause: <one sentence>\nActions: 1) <action> 2) <action> 3) <action>")

# Llama 3.x end-of-answer markers. The runner may not stop on them by itself,
# so we watch the output and stop as soon as one appears.
STOP = ("<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>")

NICE = {"hole_burst": "hole / burst", "stain_oil": "oil stain", "broken_yarn_needle": "broken yarn / needle break",
        "lycra_run": "lycra run", "jute": "jute contamination", "unclassified_defect": "unclassified defect"}


def build_prompt(defect, severity, size_pct, decision, fabric="woven cotton"):
    user = (f"Fabric: {fabric}. Defect: {NICE.get(defect, defect)}. Severity: {severity}. "
            f"Defect size: {size_pct:.1f}% of the inspected area. System decision: {decision}. "
            "What is the likely cause and what should the operator do?")
    # Llama 3.x chat template
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + SYSTEM + "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n" + user + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")


class LlamaAdvisor:
    """Load once, then call advise() for each defect (loading takes a few seconds)."""

    def __init__(self, model="llama/model.pte", tokenizer="llama/tokenizer.json", max_new_tokens=130):
        self.runner = TextLLMRunner(model, tokenizer)
        self.max_new_tokens = max_new_tokens
        self.cache = {}  # same defect + severity -> reuse the answer (instant)

    def advise(self, defect, severity, size_pct=0.0, decision="REJECT", use_cache=True):
        key = (defect, severity, decision)
        if use_cache and key in self.cache:
            return self.cache[key], 0.0
        pieces = []

        def on_token(piece):
            pieces.append(piece)
            if any(m in "".join(pieces[-4:]) for m in STOP):
                self.runner.stop()

        cfg = GenerationConfig(max_new_tokens=self.max_new_tokens, temperature=0.0, echo=False, seq_len=512)
        t = time.time()
        self.runner.reset()
        self.runner.generate(build_prompt(defect, severity, size_pct, decision), cfg, token_callback=on_token)
        text = "".join(pieces)
        for m in STOP:
            text = text.split(m)[0]
        text = text.strip()
        self.cache[key] = text
        return text, time.time() - t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama/model.pte")
    ap.add_argument("--tokenizer", default="llama/tokenizer.json")
    ap.add_argument("--defect", default="hole_burst")
    ap.add_argument("--severity", default="High")
    ap.add_argument("--size", type=float, default=6.1)
    ap.add_argument("--decision", default="REJECT")
    a = ap.parse_args()

    t0 = time.time()
    adv = LlamaAdvisor(a.model, a.tokenizer)
    print(f"Llama loaded in {time.time() - t0:.1f} s")
    text, secs = adv.advise(a.defect, a.severity, a.size, a.decision)
    print("\n=== Llama 3.2 advice ===\n" + text)
    print(f"\n(generated in {secs:.1f} s, fully offline on this device)")
