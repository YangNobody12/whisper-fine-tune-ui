import gradio as gr
import os
import random
import pandas as pd
import librosa
import json
import numpy as np
import torch
import subprocess
import sys
import re

import ast

from dataset import load_and_prepare_dataset

# -------------------------------
# Device select
# -------------------------------
device = (
    "cuda"
    if torch.cuda.is_available()
    else (
        "xpu"
        if torch.xpu.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
)

# cache dataset and processor
dataset_cache = None
processor_cache = None
tokenizer_cache = None
feature_extractor_cache = None

# global process
training_process = None


# -------------------------------
# Project management
# -------------------------------
def create_project(project_name: str):
    path = os.path.join("dataset", project_name)
    os.makedirs(os.path.join(path, "wavs"), exist_ok=True)
    return f"✅ Created project: {project_name}"


def get_projects():
    if not os.path.exists("dataset"):
        os.makedirs("dataset")
    list_name_project = os.listdir("dataset")
    return [
        name
        for name in list_name_project
        if os.path.isdir(os.path.join("dataset", name))
    ]


def check_project_files(project_name: str):
    if not project_name:
        return "❌ No project selected"
    path = os.path.join("dataset", project_name)
    path_wavs = os.path.join(path, "wavs")
    if not os.path.exists(path):
        return f"❌ Project '{project_name}' does not exist"

    audio_files = []
    if os.path.exists(path_wavs):
        audio_files = [f for f in os.listdir(path_wavs) if f.endswith((".wav", ".mp3"))]
    metadata_files = [
        f
        for f in os.listdir(path)
        if f.startswith("metadata") and f.endswith((".json", ".csv"))
    ]

    report = []
    report.append(f"📂 Checking project: {project_name}")
    report.append(f"🎵 Audio files found: {len(audio_files)}")
    if audio_files:
        report.extend([f"   - {f}" for f in audio_files[:5]])

    if metadata_files:
        report.append(f"📝 Metadata file(s) found: {', '.join(metadata_files)}")
    else:
        report.append("⚠️ No metadata file found")

    return "\n".join(report)


def get_random_audio(project_name: str):
    path = os.path.join("dataset", project_name, "wavs")
    if not os.path.exists(path):
        return None, f"❌ No 'wavs' folder in {project_name}"

    files = [f for f in os.listdir(path) if f.endswith((".wav", ".mp3"))]
    if not files:
        return None, "⚠️ No audio files found"

    random_file = random.choice(files)
    return os.path.join(path, random_file), f"🎵 Selected: {random_file}"


# -------------------------------
# Dataset preparation
# -------------------------------
def prepare_dataset_ui(metadata_path):
    global dataset_cache, processor_cache, tokenizer_cache, feature_extractor_cache
    dataset_cache, processor_cache, tokenizer_cache, feature_extractor_cache = (
        load_and_prepare_dataset(metadata_path)
    )
    return "✅ Dataset prepared!"


def prepare_from_project(projech_name):
    path_csv = os.path.join("dataset", projech_name, "metadata.csv")
    df = pd.read_csv(path_csv, sep="|")

    audio_dataset = []
    for idx, row in df.iterrows():
        audio_id = str(row[0]).strip()
        transcription = row[1]
        file_path = os.path.join("dataset", projech_name, "wavs", audio_id + ".wav")
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
        try:
            audio_array, sr = librosa.load(file_path, sr=16000)
            audio_array = np.array(audio_array, dtype=np.float32)
            data = {
                "path": str(file_path).replace("\\", "/"),
                "transcription": transcription,
                "audio": {
                    "array": audio_array.tolist(),
                    "sampling_rate": sr,
                    "path": str(file_path).replace("\\", "/"),
                },
            }
            audio_dataset.append(data)
        except Exception as e:
            print(f"❌ Error loading {audio_id}: {e}")

    json_path = f"dataset/{projech_name}/metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audio_dataset, f, ensure_ascii=False, indent=2)

    return "Dataset prepared successfully!"


# -------------------------------
# Metrics Parser
# -------------------------------
def parse_metrics(line):
    metrics = {}
    try:
        step_match = re.search(r"step\s*=\s*(\d+)", line)
        if step_match:
            metrics["step"] = int(step_match.group(1))

        loss_match = re.search(r"loss\s*=\s*([\d\.]+)", line)
        if loss_match:
            metrics["train_loss"] = float(loss_match.group(1))

        eval_loss_match = re.search(r"eval_loss\s*=\s*([\d\.]+)", line)
        if eval_loss_match:
            metrics["eval_loss"] = float(eval_loss_match.group(1))

        wer_match = re.search(r"wer\s*=\s*([\d\.]+)", line)
        if wer_match:
            metrics["wer"] = float(wer_match.group(1))
    except Exception:
        pass
    return metrics


# -------------------------------
# Training control
# -------------------------------
def train_ui(
    project_name,
    output_dir,
    max_steps,
    lr,
    model_name,
    language,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    save_steps,
    eval_steps,
):
    global training_process
    if not project_name:
        yield "❌ Please select a project", pd.DataFrame()
        return

    script_path = os.path.abspath("train_cli.py")
    cmd = [
        sys.executable,
        script_path,
        "--project",
        project_name,
        "--output_dir",
        output_dir,
        "--max_steps",
        str(max_steps),
        "--learning_rate",
        str(lr),
        "--model",
        model_name,
        "--language",
        language,
        "--per_device_train_batch_size",
        str(per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(per_device_eval_batch_size),
        "--save_steps",
        str(save_steps),
        "--eval_steps",
        str(eval_steps),
    ]

    yield f"Running: {' '.join(cmd)}", pd.DataFrame()

    training_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=os.getcwd(),
        encoding="utf-8",
        errors="replace",
    )

    metrics_list = []

    for line in iter(training_process.stdout.readline, ""):
        if not line.strip():
            continue

        # ✅ แสดง log ที่ CMD
        print(line.strip(), flush=True)

        parsed_metrics = None
        if line.strip().startswith("{") and line.strip().endswith("}"):
            try:
                parsed_metrics = ast.literal_eval(line.strip())
            except Exception:
                pass

        if parsed_metrics:
            metrics = {
                "step": parsed_metrics.get("step"),
                # "train_loss": parsed_metrics.get("train_loss"),
                "eval_loss": parsed_metrics.get("eval_loss"),
                "wer": parsed_metrics.get("eval_wer"),
                "epoch": parsed_metrics.get("epoch"),
            }
            metrics_list.append(metrics)
            df = pd.DataFrame(metrics_list)
            yield line.strip(), df
        else:
            yield line.strip(), pd.DataFrame(metrics_list)


    training_process.stdout.close()
    training_process.wait()

    rc = training_process.returncode
    training_process = None
    if rc == 0:
        yield "✅ Training finished!", pd.DataFrame(metrics_list)
    else:
        yield f"❌ Training failed with exit code {rc}", pd.DataFrame(metrics_list)


def stop_training():
    global training_process
    if training_process is not None and training_process.poll() is None:
        training_process.terminate()
        try:
            training_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            training_process.kill()
        training_process = None
        return "🛑 Training stopped by user!"
    else:
        return "⚠️ No training process is running."


# -------------------------------
# Load & Save Settings
# -------------------------------
def load_settings(project_name):
    default_settings = {
        "model": "openai/whisper-small",
        "language": "english",
        "output_dir": f"./whisper-finetuned-{project_name}",
        "max_steps": 5000,
        "save_steps": 1000,
        "eval_steps": 1000,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "learning_rate": 1e-5,
    }
    settings_path = os.path.join("dataset", project_name, "settings.json")
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            return {**default_settings, **settings}
        except Exception as e:
            print(f"⚠️ Error reading settings.json: {e}")
    return default_settings


def save_settings(
    project_name,
    model,
    language,
    output_dir,
    max_steps,
    save_steps,
    eval_steps,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    learning_rate,
):
    settings_path = os.path.join("dataset", project_name, "settings.json")
    settings = {
        "model": model,
        "language": language,
        "output_dir": output_dir,
        "max_steps": int(max_steps),
        "save_steps": int(save_steps),
        "eval_steps": int(eval_steps),
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "per_device_eval_batch_size": int(per_device_eval_batch_size),
        "learning_rate": float(learning_rate),
    }
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
    return f"✅ Saved settings to {settings_path}"


def setup_training_settings(project_name):
    if not project_name:
        return (
            "./whisper-finetuned",
            5000,
            1000,
            1000,
            16,
            16,
            1e-5,
            "openai/whisper-small",
            "english",
        )
    settings = load_settings(project_name)
    return (
        settings["output_dir"],
        settings["max_steps"],
        settings["save_steps"],
        settings["eval_steps"],
        settings["per_device_train_batch_size"],
        settings["per_device_eval_batch_size"],
        settings["learning_rate"],
        settings["model"],
        settings["language"],
    )


# -------------------------------
# Build UI
# -------------------------------
def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 🎙️ Whisper Fine-tune UI")

        with gr.Tab("Prepare Dataset"):
            gr.Markdown("### Project Management")
            with gr.Row():
                project_name = gr.Textbox(
                    label="Project Name", placeholder="Enter project name"
                )
                status_box = gr.Textbox(label="Status", placeholder="Status")
                create_project_button = gr.Button("Create Project")
                create_project_button.click(
                    create_project, inputs=project_name, outputs=status_box
                )

            gr.Markdown("### Select Project")
            project_dropdown = gr.Dropdown(
                get_projects(), label="Project Name", interactive=True
            )
            refresh_btn = gr.Button("🔄 Refresh")
            refresh_btn.click(
                lambda: gr.update(choices=get_projects()), outputs=project_dropdown
            )

            gr.Markdown("### Check Files in Project")
            check_output = gr.Textbox(label="Check Status", lines=8)
            check_btn = gr.Button("Check")
            check_btn.click(
                check_project_files, inputs=project_dropdown, outputs=check_output
            )

            gr.Markdown("### Random Audio from Project (wavs/)")
            random_audio_output = gr.Audio(label="Random Audio", type="filepath")
            random_audio_status = gr.Textbox(label="Status")
            random_btn = gr.Button("Get Random Audio")
            random_btn.click(
                get_random_audio,
                inputs=project_dropdown,
                outputs=[random_audio_output, random_audio_status],
            )

            gr.Markdown("### Prepare Dataset for Training")
            dataset_btn = gr.Button("Prepare Dataset")
            dataset_output = gr.Textbox(label="Prepare Status")
            dataset_btn.click(
                prepare_from_project,
                inputs=project_dropdown,
                outputs=dataset_output,
                show_progress=True,
            )

        with gr.Tab("Train Model"):
            gr.Markdown("### Train Whisper Model")
            with gr.Row():
                project_dropdown2 = gr.Dropdown(
                    os.listdir("dataset"), label="Select Project", interactive=True
                )
                output_dir = gr.Textbox(label="Output Directory")
                model_name = gr.Textbox(
                    label="Model Checkpoint", value="openai/whisper-small"
                )
                language = gr.Textbox(label="Language", value="english")

            with gr.Row():
                per_device_train_batch_size = gr.Number(
                    label="Train Batch Size", value=16
                )
                per_device_eval_batch_size = gr.Number(
                    label="Eval Batch Size", value=16
                )
                max_steps = gr.Number(label="Max Steps", value=5000)

            with gr.Row():
                save_steps = gr.Number(label="Save Steps", value=1000)
                eval_steps = gr.Number(label="Eval Steps", value=1000)
                lr = gr.Number(label="Learning Rate", value=1e-5)

            setup_btn = gr.Button("🔧 Auto Setup")
            setup_btn.click(
                setup_training_settings,
                inputs=project_dropdown2,
                outputs=[
                    output_dir,
                    max_steps,
                    save_steps,
                    eval_steps,
                    per_device_train_batch_size,
                    per_device_eval_batch_size,
                    lr,
                    model_name,
                    language,
                ],
            )

            gr.Markdown("### Training Control")
            train_btn = gr.Button("Start Training")
            stop_btn = gr.Button("Stop Training")

            train_output = gr.Textbox(label="Training Log", lines=20)
            metrics_output = gr.Dataframe(
                headers=["step", "train_loss", "eval_loss", "wer"], row_count=5
            )

            train_btn.click(
                train_ui,
                inputs=[
                    project_dropdown2,
                    output_dir,
                    max_steps,
                    lr,
                    model_name,
                    language,
                    per_device_train_batch_size,
                    per_device_eval_batch_size,
                    save_steps,
                    eval_steps,
                ],
                outputs=[train_output, metrics_output],
            )
            stop_btn.click(stop_training, outputs=train_output)

        with gr.Tab("Inference"):
            gr.Markdown("### Inference (placeholder)")
            with gr.Row():
                gr.Textbox(label="Input Audio (path)")
                gr.Textbox(label="Transcript")

    return demo


demo = create_ui()

if __name__ == "__main__":
    demo.launch(show_error=True, share=True)
