from transformers import AutoTokenizer
import random

# taking a pretrained tokenizer for simplicity - one which has already been trained for German <-> English translation
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

# add special tokens for language direction
special_tokens = {"direction_tokens": ["<2en>", "<2de>"]}
tokenizer.add_special_tokens(special_tokens)

# unidirectional tokenization function
def tokenize_func(batch):
    return tokenizer(
        batch["translation"]["de"],
        text_target=batch["translation"]["en"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

# bidirectional tokenization function
def tokenize_func_bidirectional(batch):
    
    sources = []
    targets = []
    for ex in batch["translation"]:
        if random.random() > 0.5:
            # German to English
            sources.append("<2en> " + ex["de"])
            targets.append(ex["en"])
        else:
            # English to German
            sources.append("<2de> " + ex["en"])
            targets.append(ex["de"])

    model_inputs = tokenizer(
        sources,
        text_target=targets,
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    return model_inputs

