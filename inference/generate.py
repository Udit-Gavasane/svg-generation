import os
import re
import time
import warnings
import torch
import pandas as pd
import xml.etree.ElementTree as ET
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import transformers

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
ADAPTER_PATH = r"svg_3b_adapter"
TEST_PATH = r"data\test.csv"
OUTPUT_PATH = r"submission.csv"

SYSTEM_PROMPT = "You are an SVG code generator. Given a description, output valid SVG code only. No explanations."

SVG_REGEX = re.compile(r"<svg[\s\S]*?</svg>", flags=re.IGNORECASE)

FALLBACK = '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256"><rect width="256" height="256" fill="white"/><circle cx="128" cy="128" r="64" fill="black"/></svg>'

DISALLOWED = ['feGaussianBlur', 'feBlend', 'feColorMatrix', 'feComposite',
              'feFlood', 'feMerge', 'feMergeNode', 'feOffset', 'feDropShadow',
              'feTurbulence', 'feDisplacementMap', 'feConvolveMatrix',
              'feDiffuseLighting', 'feSpecularLighting', 'fePointLight',
              'feDistantLight', 'feSpotLight', 'feMorphology', 'feTile',
              'feImage', 'feComponentTransfer', 'feFuncR', 'feFuncG',
              'feFuncB', 'feFuncA', 'animate', 'animateTransform',
              'animateMotion', 'set', 'foreignObject', 'script', 'image']

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Loading adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
print(f"✅ Model loaded on {next(model.parameters()).device}")
transformers.logging.set_verbosity_error()


def fix_svg(svg):
    svg = re.sub(r'width="[^"]*"', 'width="256"', svg, count=1)
    svg = re.sub(r'height="[^"]*"', 'height="256"', svg, count=1)
    svg = re.sub(r"width='[^']*'", "width='256'", svg, count=1)
    svg = re.sub(r"height='[^']*'", "height='256'", svg, count=1)
    svg = re.sub(r'viewBox="[^"]*"', 'viewBox="0 0 256 256"', svg, count=1)
    if 'viewBox' not in svg:
        svg = svg.replace('<svg ', '<svg viewBox="0 0 256 256" ', 1)
    if 'xmlns' not in svg:
        svg = svg.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    for tag in DISALLOWED:
        svg = re.sub(rf'<{tag}[^>]*/>', '', svg, flags=re.IGNORECASE)
        svg = re.sub(rf'<{tag}[^>]*>[\s\S]*?</{tag}>', '', svg, flags=re.IGNORECASE)
    # Enforce max 8000 chars
    if len(svg) > 8000:
        svg = svg[:7900] + '</svg>'
    return svg


def is_valid(svg):
    if not svg:
        return False
    try:
        root = ET.fromstring(svg)
        tag = root.tag.split('}')[-1] if '}' in root.tag else root.tag
        return tag == 'svg'
    except:
        return False


def generate_svg(prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=800,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    svg = SVG_REGEX.search(raw)
    svg = svg.group(0).strip() if svg else ""

    if not svg and '<svg' in raw:
        start = raw.index('<svg')
        svg = raw[start:]
        # Remove last incomplete tag
        svg = re.sub(r'<[^>]*$', '', svg)
        # Remove incomplete path at end
        svg = re.sub(r'<path[^>]*$', '', svg)
        svg += '</svg>'

    if svg:
        svg = fix_svg(svg)

    if not is_valid(svg):
        return FALLBACK

    return svg


test_df = pd.read_csv(TEST_PATH)
rows = []
fallbacks = 0
t0 = time.time()

for i, row in test_df.iterrows():
    svg = generate_svg(row["prompt"])
    if svg == FALLBACK:
        fallbacks += 1
    rows.append({"id": row["id"], "svg": svg})

    elapsed = time.time() - t0
    rate = (i + 1) / elapsed
    remaining = (len(test_df) - i - 1) / rate
    print(f"[{i+1}/1000] fallbacks: {fallbacks} | {elapsed/(i+1):.1f}s/sample | ~{remaining/60:.1f}min left")

sub = pd.DataFrame(rows)
sub.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Done in {(time.time()-t0)/60:.1f}min | fallbacks: {fallbacks}/1000")
print(f"Saved to {OUTPUT_PATH}")
