# Whisper Fine-tune UI & CLI

โครงการนี้เป็นเครื่องมือสำหรับการ **Fine-tune โมเดล Whisper** โดยรองรับการใช้งานผ่าน:
- Command Line Interface (CLI)
- Web UI (Gradio)

## โครงสร้างโปรเจกต์
```
ui_train_whisper/
├── finetune_gradio.py               # Gradio UI สำหรับ train + dataset prepare
├── train.py             # ฟังก์ชัน train_model + custom callback
├── train_cli.py         # CLI สำหรับ train จาก command line
├── dataset.py           # ฟังก์ชัน load และ prepare dataset
├── dataset/             # เก็บโปรเจกต์, metadata, และไฟล์เสียง
│   └── <project_name>/
│       ├── wavs/        # ไฟล์เสียง .wav / .mp3
│       ├── metadata.csv # transcription
│       └── metadata.json
└── README.md
```

---

## การติดตั้ง

```bash
# สร้าง environment ใหม่ (แนะนำ conda)
conda create -n whisper_env python=3.10 -y
conda activate whisper_env

# ติดตั้ง dependencies
pip install -r requirements.txt

pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

```

**requirements.txt (ตัวอย่าง):**
```
torch
transformers
datasets
evaluate
librosa
pandas
numpy
gradio
```

---

## การใช้งาน

### 1) เตรียม Dataset
สร้างโปรเจกต์ใหม่:
```bash
python finetune_gradio.py
```
ไปที่แท็บ **Prepare Dataset**
- ใส่ชื่อโปรเจกต์ เช่น `my_project`
- อัปโหลดไฟล์เสียงไว้ใน `dataset/my_project/wavs/`
- สร้าง `metadata.csv` (รูปแบบ: `id|transcription`)
- กด **Prepare Dataset** → ระบบจะสร้าง `metadata.json`

---

### 2) Train ผ่าน CLI
```bash
python train_cli.py \
  --project my_project \
  --model openai/whisper-small \
  --language english \
  --output_dir ./out_model \
  --max_steps 2000 \
  --save_steps 500 \
  --eval_steps 500 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 5e-5
```

---

### 3) Train ผ่าน UI
```bash
python finetune_gradio.py
```
- เปิดเบราว์เซอร์: `http://127.0.0.1:7860`
- เลือกโปรเจกต์
- กด **Auto Setup** เพื่อโหลด config
- กด **Start Training**

UI แสดง:
- Training log (stdout)
- Metrics table: `step, train_loss, eval_loss, wer, epoch`

---

## Metrics Logging
ระหว่างการ train ระบบจะส่งออก log ในรูปแบบ JSON:
```json
{"step": 15, "epoch": 1.07, "eval_loss": 5.18, "wer": 120.38}
```

ซึ่ง UI จะ parse และแสดงในตารางอัตโนมัติ

---

##  รองรับหลาย GPU
ใช้ HuggingFace `Trainer` → ระบบตรวจจับ GPU อัตโนมัติ  
เช่น:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 train_cli.py --project my_project ...
```

---

##  TODO
- [ ] เพิ่ม Inference demo ใน UI
- [ ] รองรับการ upload dataset จากหน้าเว็บ
- [ ] Export log เป็น CSV

---

##  License
MIT License

