from transformers import AutoTokenizer
import random

# unidirectional tokenization function
def tokenize_func(batch, tokenizer):
    return tokenizer(
        batch["translation"]["de"],
        text_target=batch["translation"]["en"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

# bidirectional tokenization function
def tokenize_func_bidirectional(batch, tokenizer):
    
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

