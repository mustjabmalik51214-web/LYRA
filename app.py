import io
import json
import re
import zipfile
from flask import Flask, jsonify, render_template, request, send_file
import torch
from transformers import pipeline

app = Flask(__name__)

print("Loading Qwen1.5 0.5B Chat Model...")
pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen1.5-0.5B-Chat",
    torch_dtype=torch.float32,
    device_map="auto"
)
print("Model Loaded Successfully!")

SYSTEM_PROMPT = """You are LYRAMOON, an expert female AI software engineer created by MUHAMMAD TAQI.
Your task is to write clean, complete HTML/CSS/JS code in a single snippet inside ```html code block.

Rules:
1. Always wrap the complete code inside a ```html block.
2. If previous code is provided, update and improve it according to user instruction instead of starting from scratch.
3. Include inline CSS and JavaScript inside the HTML output.
"""

def extract_code(text):
    # Match markdown code blocks ```html ... ``` or ``` ... ```
    match = re.search(r"```(?:html)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json or {}
        user_prompt = data.get("prompt", "")
        current_code = data.get("current_code", "")

        if not user_prompt:
            return jsonify({"success": False, "response": "Please enter a prompt."}), 400

        full_user_content = f"Instruction: {user_prompt}"
        if current_code:
            full_user_content += f"\n\nCurrent Code:\n```html\n{current_code}\n```\nUpdate this existing code according to instruction."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_user_content}
        ]
        
        prompt = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = pipe(
            prompt, 
            max_new_tokens=1024, 
            do_sample=True, 
            temperature=0.4, 
            top_k=40, 
            top_p=0.9
        )
        
        generated_text = outputs[0]["generated_text"]
        
        if "<|im_start|>assistant" in generated_text:
            raw_response = generated_text.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
        else:
            raw_response = generated_text.strip()

        # Clean code extraction using regex
        final_code = extract_code(raw_response)

        return jsonify({
            "success": True,
            "explanation": "Code generated/updated successfully!",
            "code": final_code
        })

    except Exception as e:
        return jsonify({"success": False, "response": f"Error: {str(e)}"}), 500

@app.route("/download", methods=["POST"])
def download():
    try:
        data = request.json or {}
        code = data.get("code", "")
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr('index.html', code)
        
        zip_buffer.seek(0)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='TAQI.zip'
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
