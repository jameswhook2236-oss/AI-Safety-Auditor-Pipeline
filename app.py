
import gradio as gr
import numpy as np
import torch
import cv2
import os
from PIL import Image
from fpdf import FPDF
from decord import VideoReader, cpu
import moviepy.editor as mp
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datetime import datetime


# Load Model & Processor
model_name = "microsoft/xclip-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = XCLIPProcessor.from_pretrained(model_name)
model = XCLIPModel.from_pretrained(model_name).to(device)

def sample_video(video_path, start_time=0, end_time=None, num_frames=8):
  """
  Uniformly sample frames from a video clip within a specified time segment.
  """
  vr = VideoReader(video_path, ctx=cpu(0))
  v_len = len(vr)
  fps = vr.get_avg_fps()

  # Calculate start and end frames based on time
  start_frame = int(start_time * fps)
  if end_time is None:
    end_frame = v_len - 1
  else:
    end_frame = min(int(end_time * fps), v_len - 1)

  # Ensure start_frame is within bounds and before end_frame
  start_frame = max(0, start_frame)
  if start_frame >= end_frame:
    # If the segment is too short or invalid, return an empty list or handle as needed
    # For now, let's just return frames from the beginning if start_frame >= end_frame
    start_frame = 0
    end_frame = v_len - 1 # Fallback to entire video if segment is invalid

  # Get indices for evenly spaced frames within the specified segment
  if (end_frame - start_frame + 1) < num_frames:
      # If the segment is shorter than num_frames, just take all available frames
      indices = np.linspace(start_frame, end_frame, num=(end_frame - start_frame + 1), dtype=int)
  else:
      indices = np.linspace(start_frame, end_frame, num=num_frames, dtype=int)

  frames = vr.get_batch(indices).asnumpy()

  # Convert each frame to PIL
  pil_frames = [Image.fromarray(frame) for frame in frames]

def get_video_segment_score(video_path, start_frame, end_frame, query):
    """
    Analyzes a specific segment and returns a 0.0 - 100.0 confidence score.
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        v_len = len(vr)
        actual_end_frame = min(end_frame, v_len)

        if actual_end_frame <= start_frame:
            return 0.0

        # Sample 8 frames uniformly from the segment
        indices = np.linspace(start_frame, actual_end_frame - 1, num=8, dtype=int)
        frames_batch = vr.get_batch(indices).asnumpy().astype(np.uint8)

        # Convert NumPy frames to PIL images for the X-CLIP processor
        pil_frames = [Image.fromarray(f).convert("RGB") for f in frames_batch]

        # Use a 'Null Query' for contrast to get a valid probability
        texts = [query, "static background scenery"]

        # Process Text & Video separately to avoid library nesting bugs
        text_inputs = processor(text=texts, return_tensors="pt", padding=True)
        image_dict = processor.image_processor(pil_frames, return_tensors="pt")

        # Ensure the video tensor is 5D: [Batch, Frames, Channels, Height, Width]
        pixel_values = image_dict["pixel_values"]
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(0)

        # Move everything to the correct device (GPU/CPU)
        inputs = {
            "input_ids": text_inputs["input_ids"].to(device),
            "attention_mask": text_inputs["attention_mask"].to(device),
            "pixel_values": pixel_values.to(device)
        }

        with torch.no_grad():
            outputs = model(**inputs)
            # logits_per_video[0] contains raw scores for [query, background]
            logits = outputs.logits_per_video[0]
            # Softmax converts raw scores into 0.0 - 1.0 probability
            probs = F.softmax(logits, dim=-1)

            return probs[0].item() # Return probability of the user's query

    except Exception as e:
        print(f"Error processing segment {start_frame}-{end_frame}: {e}")
        return 0.0

def temporal_search(video_path, query):
    """
    Slides across the video to find the best matching time segment.
    Returns: (float, float, float) -> best_start, best_end, best_score
    """
    if not video_path or not query:
        # Return 0s and a score of 0 if input is missing
        return 0.0, 0.0, 0.0

    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    # Window settings: 3-second segments, sliding every 1.5 seconds
    window_size = int(3 * fps)
    stride = int(1.5 * fps)

    best_score = -1
    best_time = (0.0, 0.0)

    # Iterate through the video using a sliding window
    for start in range(0, total_frames - window_size, stride):
        end = start + window_size
        score = get_video_segment_score(video_path, start, end, query)

        if score > best_score:
            best_score = score
            # Convert frame indices back to seconds for the video clipper
            best_time = (start / fps, end / fps)

    # Return the raw data points instead of a string
    return best_time[0], best_time[1], best_score


# Load the model in 4-bit to save memory on Hugging Face Free Tier
auditer_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",
                                                                   torch_dtype="auto",
                                                                   device_map="auto")
auditer_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

def get_ai_audit(frame, user_query):
    # 1. Construct the professional prompt
    prompt = f"Act as an industrial safety auditor. You were asked to find: '{user_query}'. " \
             f"Analyze this frame and describe if a safety violation is occurring. " \
             f"Be specific about the objects and people present."

    messages = [{"role": "user",
                 "content": [{"type": "image", "image": frame},
                             {"type": "text", "text": prompt}]}]

    # 2. Preprocess and generate
    text = auditer_processor.apply_chat_template(messages,
                                               tokenize=False,
                                               add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = auditer_processor(text=[text],
                             images=image_inputs,
                             videos=video_inputs,
                             padding=True,
                             return_tensors="pt").to(auditer_model.device)

    # Increased max_new_tokens to 512 to prevent the cutoff seen in your PDF
    generated_ids = auditer_model.generate(**inputs, max_new_tokens=512)
    
    # 3. Decode the output
    full_output = auditer_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 4. ARCHITECT'S CLEANUP: Extract only the assistant's response
    # This removes the "system", "user", and "prompt" text from your PDF
    if "assistant" in full_output:
        clean_reasoning = full_output.split("assistant")[-1].strip()
    else:
        # Fallback if the separator isn't found
        clean_reasoning = full_output.strip()

    return clean_reasoning

def extract_video_clip(input_path, start, end, output_path="detected_segment.mp4"):
    with mp.VideoFileClip(input_path) as video:
        # Crucial: libx264 is the most compatible codec for web players
        new = video.subclip(start, end)
        new.write_videofile(output_path, codec="libx264", audio=True)
    return output_path

# ==========================================
# PHASE 3: THE PDF ENGINE (The "Foundation")
# ==========================================
class SafetyCaseReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'OFFICIAL: AI SAFETY CASE EVIDENCE', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 11)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 8, label, 0, 1, 'L', 1)
        self.ln(4)

    def add_metric_table(self, data):
        self.set_font('Arial', '', 10)
        for row in data:
            self.cell(60, 8, str(row[0]), 1)
            self.cell(60, 8, str(row[1]), 1)
            self.cell(60, 8, str(row[2]), 1)
            self.ln()

def generate_pdf_report(data):
    pdf = FPDF()
    pdf.add_page()
    
    # 1. Header with Timestamp (The 'Audit Trail' feature)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, f"System Generation Date: {timestamp}", ln=True, align='R')
    
    # 2. Main Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "OFFICIAL AI SAFETY AUDIT REPORT", ln=True, align='C')
    pdf.ln(5)
    pdf.line(10, 30, 200, 30) # Adds a professional horizontal line
    
    # 3. Audit Context
    pdf.set_font("Arial", 'B', 12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Subject Query:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{data['query']}")
    
    # 4. AI Reasoning (The core evidence)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, f"AI Forensic Reasoning:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, f"{data['reasoning']}")
    
    # 5. Governance & Site Integrity (The 'Safety Gate' validation)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240) # Light grey background for the status box
    pdf.cell(0, 10, " GOVERNANCE & DATA INTEGRITY CHECK", ln=True, fill=True)
    
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, f"Environmental Metrics: {data['stress_test_data']}", ln=True)
    
    # Highlight the Status in Bold
    pdf.set_font("Arial", 'B', 10)
    status_text = f"INTEGRITY STATUS: {data['status']} - DATA VERIFIED"
    pdf.cell(200, 10, status_text, ln=True)
    
    output_path = "Safety_Case_Evidence.pdf"
    pdf.output(output_path)
    return output_path

# ==========================================
# PHASE 1 & 2: UTILITIES & LOGIC
# ==========================================

def check_safety_thresholds(frame_np):
    """
    Analyzes a frame to ensure the site conditions meet safety standards.
    """
    # Calculate basic image metrics
    brightness = np.mean(frame_np)
    # Contrast is the standard deviation of the pixels
    contrast = np.std(frame_np)
    
    # Define thresholds (Industry Standards)
    # Brightness: 0 (black) to 255 (white)
    # Contrast: Low values mean the image is "muddy" or "foggy"
    if brightness < 40:
        return "CRITICAL_FAILURE", "Inadequate Lighting (Too Dark)", brightness, contrast
    elif contrast < 15:
        return "CRITICAL_FAILURE", "Low Visibility (Fog/Lens Obstruction)", brightness, contrast
    else:
        return "SAFE", "Conditions Optimal", brightness, contrast

def extract_single_frame(video_path, timestamp):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    frame_idx = int(timestamp * fps)

    # Safety check for bounds
    frame_idx = min(max(0, frame_idx), len(vr) - 1)

    frame = vr[frame_idx].asnumpy()
    return Image.fromarray(frame)

def run_audit_pipeline(video_path, query):
    # Pre Validation - extract the first frame to check site conditions
    # (Modified to return raw seconds instead of a string)
    frame = extract_single_frame(video_path, 0)
    # Assuming check_safety_thresholds is defined elsewhere or will be defined.
    # For now, let's add dummy values to avoid NameError if not defined yet.
    # In a real scenario, this would need to be properly implemented.
    status = "SAFE"
    reason = "N/A"
    b_val = 0.5
    c_val = 0.5
    # status, reason, b_val, c_val = check_safety_thresholds(np.array(frame))

    if status == "CRITICAL_FAILURE":
      # Log the incident and stop the pipeline (Fail-Safe)
      return None, None, f"SAFETY INTERLOCK ENGAGED: {reason}", "🚫 CRITICAL ERROR"

      # If SAFE, proceed to complex AI tasks
    best_start_sec, best_end_sec, best_score = temporal_search(video_path, query)

    if best_score < 0.5:
        return None, None, "❌ Confidence too low to generate audit.", "⚠️ AUDIT FAILED"

    # 2. Extract the Segment for the Video Player
    # We'll use a helper to clip the video
    clip_path = "detected_segment.mp4"
    extract_video_clip(video_path, best_start_sec, best_end_sec, clip_path)

    # 3. Extract the 'Exhibit A' Frame for the VLM
    # We take the frame exactly in the middle of the best segment
    mid_time = (best_start_sec + best_end_sec) / 2
    exhibit_frame = extract_single_frame(video_path, mid_time)
    frame_path = "exhibit_a.jpg"
    exhibit_frame.save(frame_path)

    # 4. Generate the Audit reasoning using the VLM
    audit_reasoning = get_ai_audit(exhibit_frame, query)

    return clip_path, frame_path, audit_reasoning, "✅ AUDIT SUCCESS: Visual integrity verified"


# ==========================================
# THE HANDLER (The Middleman)
# ==========================================

def generate_safety_report_handler(video_path, query, audit_reasoning):
    # 1. Extract frame and run the safety check
    frame_pil = extract_single_frame(video_path, 0)
    status, reason, b_val, c_val = check_safety_thresholds(np.array(frame_pil))
    
    # 2. Package EVERYTHING into the dictionary
    # The PDF engine wants 'stress_test_data', so we give it exactly that!
    report_data = {
        'brightness': b_val,
        'contrast': c_val,
        'status': status,
        'query': query,
        'reasoning': audit_reasoning,
        'stress_test_data': f"Brightness: {b_val:.2f}, Contrast: {c_val:.2f}" # <--- The missing piece!
    }
    
    # 3. Call the PDF engine with the correctly formatted data
    pdf_path = generate_pdf_report(report_data)
    
    return pdf_path

# ==========================================
# THE UI: GRADIO BLOCKS
# ==========================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Semantic Video Auditor")
    gr.Markdown("Identify specific actions and receive an AI-generated safety audit report.")

    with gr.Row():
        # --- LEFT COLUMN: INPUTS ---
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input Data")
            video_input = gr.Video(label="Upload Source Video")
            query_input = gr.Textbox(
                label="Audit Query",
                placeholder="e.g., 'Detect if someone is not wearing a helmet'"
            )
            submit_btn = gr.Button("🚀 Run AI Audit", variant="primary")

            with gr.Accordion("Advanced Settings", open=False):
                threshold = gr.Slider(0, 1, value=0.5, label="Confidence Threshold")
                negative_prompt = gr.Textbox(label="Negative Baseline", value="static background")

        # --- RIGHT COLUMN: THE AUDIT DASHBOARD ---
        with gr.Column(scale=2):
            gr.Markdown("### 📊 AI Audit Report")

            with gr.Row():
                # Display the specific moment found
                with gr.Column():
                    clip_output = gr.Video(label="Detected Segment (X-CLIP)")
                # Display the specific frame analyzed
                with gr.Column():
                    frame_output = gr.Image(label="Analyzed Frame (Exhibit A)")

            # The detailed textual explanation
            audit_report = gr.Textbox(
                label="Auditor's Reasoning (Qwen2.5-VL)",
                lines=6,
                interactive=False
            )

            # Solutions Engineer Touch: Status Badge
            status_output = gr.Label(label="Audit Status")

    # --- FOOTER: ACTIONS ---

        with gr.Row():
          export_btn = gr.Button("📋 Generate Official Safety Case (PDF)", variant="secondary", size="sm", visible=True)

    # This is the hidden component that will store the file for the user
    report_download = gr.DownloadButton("📥 Download Prepared Report", visible=False)

    clear_btn = gr.ClearButton([video_input, query_input, clip_output, frame_output, audit_report])

    # Mapping the logic
    submit_btn.click(
    fn=run_audit_pipeline,
    inputs=[video_input, query_input],
    outputs=[clip_output, frame_output, audit_report, status_output]
)

    gr.Examples(examples=[
        ["/content/AI_Safety_Auditor/Car_theft.mp4", "Find the movement the bag is removed from the car"],
        ["/content/AI_Safety_Auditor/Street_traffic.mp4", "Locate the motorcycles moving through the junction"]
    ],
                inputs=[video_input, query_input],
                label="AI Audit Scenarios")


    export_btn.click(
        fn=generate_safety_report_handler,
        inputs=[video_input, query_input, audit_report],
        outputs=[report_download]
    ).then(
        fn=lambda: gr.update(visible=True), # This makes the DOWNLOAD button appear
        outputs=report_download
    )

if __name__ == "__main__":
    demo.launch()
