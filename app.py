import io
import json
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
Your job is to generate and update web code (HTML, CSS, JavaScript, or Python).

CRITICAL INSTRUCTIONS:
1. Always respond ONLY in a valid JSON format with three keys:
   - "explanation": Short summary of what you did/changed.
   - "language": "html" or "python"
   - "code": The full code output (Do not use Markdown block ``` html inside JSON).
2. If the user provides PREVIOUS CODE, you MUST UPDATE and enhance the existing code rather than creating a whole new separate project from scratch, unless explicitly told to start new.
3. Make sure the HTML/CSS/JS code is clean, production-ready, interactive, and beautifully styled.
"""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    user_prompt = data.get("prompt", "")
    current_code = data.get("current_code", "")

    if not user_prompt:
        return jsonify({"response": "Please enter a message."}), 400

    full_user_content = f"Request: {user_prompt}"
    if current_code:
        full_user_content += f"\n\nExisting Code to Update/Modify:\n{current_code}"

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
        temperature=0.3, 
        top_k=50, 
        top_p=0.95
    )
    
    generated_text = outputs[0]["generated_text"]
    
    if "<|im_start|>assistant" in generated_text:
        raw_response = generated_text.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
    else:
        raw_response = generated_text.strip()

    # Try parsing JSON output from model
    try:
        # Extract json if wrapped in ```json
        if "```json" in raw_response:
            raw_response = raw_response.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_response:
            raw_response = raw_response.split("```")[1].strip()

        parsed_data = json.loads(raw_response)
        return jsonify({
            "success": True,
            "explanation": parsed_data.get("explanation", "Code generated successfully."),
            "code": parsed_data.get("code", ""),
            "language": parsed_data.get("language", "html")
        })
    except Exception as e:
        # Fallback if raw text returned
        return jsonify({
            "success": True,
            "explanation": "Generated response:",
            "code": raw_response,
            "language": "html"
        })

@app.route("/download", methods=["POST"])
def download():
    data = request.json
    code = data.get("code", "")
    
    # In-memory ZIP buffer creation
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
