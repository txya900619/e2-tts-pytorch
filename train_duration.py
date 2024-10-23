from datasets import concatenate_datasets, load_dataset
from schedulefree import AdamWScheduleFree
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from e2_tts_pytorch import DurationPredictor
from e2_tts_pytorch.trainer import DurationPredictorTrainer, HFDataset

tokenizer_object = Tokenizer.from_file("tokenizer.json")
fast_tokenizer_object = PreTrainedTokenizerFast(tokenizer_object=tokenizer_object)
PreTrainedTokenizerFast
def tokenizer(text):
    ids = fast_tokenizer_object(text, return_tensors="pt", padding=True).input_ids
    ids[ids == 0] = -1
    return ids

duration_predictor = DurationPredictor(
    transformer = dict(
        dim = 384,
        depth = 8,
        heads = 6,
    ),
    text_num_embeds = fast_tokenizer_object.vocab_size,
    tokenizer=tokenizer,
)

train_dataset = concatenate_datasets(
    [
        load_dataset("formospeech/hat_tts", "sixian", split="train", num_proc=4),
        load_dataset(
            "formospeech/hakkaradio_news", "sixian", split="train", num_proc=4
        ),
        load_dataset("formospeech/hac_vocab", "sixian_e", split="train", num_proc=4),
    ]
)
train_dataset = train_dataset.rename_column("ipa", "transcript")
train_dataset = train_dataset.map(lambda x: {"transcript": x["transcript"].replace("-", "")}, num_proc=8)

train_dataset = HFDataset(hf_dataset=train_dataset)

# need to train duration predictor
optimizer = AdamWScheduleFree(duration_predictor.parameters(), lr=1e-3)

trainer = DurationPredictorTrainer(
    duration_predictor,
    optimizer,
    grad_accumulation_steps = 8,
    checkpoint_path = 'duration_predictor.pt',
    sample_rate=24000,
    log_file = 'duration_predictor.txt'
)

epochs = 30
batch_size = 16


trainer.train(train_dataset, epochs, batch_size, save_step=1000)
