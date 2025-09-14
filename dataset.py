from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="english", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="english", task="transcribe")

def load_and_prepare_dataset(metadata_path: str):
    dataset = load_dataset("json", data_files=metadata_path)
    dataset = dataset["train"].train_test_split(test_size=0.2)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    def prepare_batch(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = tokenizer(batch["transcription"]).input_ids
        return batch

    dataset = dataset.map(prepare_batch, remove_columns=dataset.column_names["train"], num_proc=1)
    return dataset, processor, tokenizer, feature_extractor
