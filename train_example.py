from datasets import concatenate_datasets, load_dataset
from schedulefree import AdamWScheduleFree
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from e2_tts_pytorch import E2TTS
from e2_tts_pytorch.trainer import E2Trainer, HFDataset

tokenizer_object = Tokenizer.from_file("tokenizer.json")
fast_tokenizer_object = PreTrainedTokenizerFast(tokenizer_object=tokenizer_object)
PreTrainedTokenizerFast
def tokenizer(text):
    ids = fast_tokenizer_object(text, return_tensors="pt", padding=True).input_ids
    ids[ids == 0] = -1
    return ids

# duration_predictor = DurationPredictor(
#     transformer = dict(
#         dim = 384,
#         depth = 8,
#         heads = 6,
#     ),
#     text_num_embeds = fast_tokenizer_object.vocab_size,
#     tokenizer=tokenizer,
# )

e2tts = E2TTS(
    # duration_predictor = duration_predictor,
    cond_drop_prob = 0.2,
    transformer = dict(
        dim = 512,
        depth = 12,
        heads = 6,
        attn_kwargs = dict(
            gate_value_heads = True,
            softclamp_logits = True,
        ),
    ),
    text_num_embeds = fast_tokenizer_object.vocab_size,
    tokenizer=tokenizer,
    mel_spec_kwargs = dict(
        filter_length = 1024,
        hop_length = 256,
        win_length = 1024,
        n_mel_channels = 100,
        sampling_rate = 24000,
    ),
    frac_lengths_mask = (0.7, 0.9)
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
optimizer = AdamWScheduleFree(e2tts.parameters(), lr=3e-4)

trainer = E2Trainer(
    e2tts,
    optimizer,
    grad_accumulation_steps = 8,
    checkpoint_path = 'e2tts.pt',
    sample_rate=24000,
    log_file = 'e2tts.txt'
)

epochs = 30
batch_size = 8


trainer.train(train_dataset, epochs, batch_size, save_step=1000)
