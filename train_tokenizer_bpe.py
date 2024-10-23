from datasets import concatenate_datasets, load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

train_dataset = concatenate_datasets(
    [
        load_dataset("formospeech/hat_tts", "sixian", split="train", num_proc=4),
        load_dataset(
            "formospeech/hakkaradio_news", "sixian", split="train", num_proc=4
        ),
        load_dataset("formospeech/hac_vocab", "sixian_e", split="train", num_proc=4),
    ]
)
train_dataset = train_dataset.remove_columns(set(train_dataset.column_names) - {"ipa"})
train_dataset = train_dataset.map(lambda x: {"ipa": x["ipa"].replace("-", "").replace(" <sil>", "")})


tokenizer = Tokenizer(BPE())
# tokenizer.pre_tokenizer = WhitespaceSplit()

trainer = BpeTrainer(vocab_size=256, special_tokens=["<pad>", "<sil>"])


tokenizer.train_from_iterator(iterator=train_dataset["ipa"], trainer=trainer)
tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
print(tokenizer.get_vocab())
output = tokenizer.encode("t͡sʰ-i_24-aŋ_24-m-un_11 <sil>".replace("-", ""))
print(output.tokens)
tokenizer.save("tokenizer.json")
