import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    TrainerCallback,
)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ✅ Callback สำหรับ log ข้อมูล training
class MetricsLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print()
            logs["step"] = state.global_step
            # logs.setdefault("train_loss", None)        # train_loss
            logs.setdefault("eval_loss", None)
            logs.setdefault("eval_wer", None)

            print(logs, flush=True)  # print ออกทั้ง cmd + ให้ UI อ่าน


def train_model(
    dataset,
    processor,
    tokenizer,
    model="openai/whisper-small",
    language="english",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    output_dir="./whisper-small-finetuned",
    max_steps=5000,
    save_steps=1000,
    eval_steps=1000,
    lr=1e-5,
):
    model = WhisperForConditionalGeneration.from_pretrained(model)
    model.generation_config.language = language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        warmup_steps=500,
        max_steps=max_steps,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[MetricsLoggerCallback()],  # ✅ เพิ่ม callback
    )

    trainer.train()
    trainer.save_model(output_dir)

    return "✅ Training completed!"
