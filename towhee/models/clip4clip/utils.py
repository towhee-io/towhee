import regex as re
import numpy as np
from typing import List
from towhee.models.clip.simple_tokenizer import SimpleTokenizer, whitespace_clean, basic_clean


def tokenize(text: str) -> List:
    """
    Use SimpleTokenizer to tokenize text.
    Args:
        text (`str`):
            Text to tokenize

    Returns:
        Tokenized infos.
    """
    tokenizer = SimpleTokenizer()
    tokens = []
    text = whitespace_clean(basic_clean(text)).lower()
    for token in re.findall(tokenizer.pat, text):
        token = "".join(tokenizer.byte_encoder[b] for b in token.encode("utf-8"))
        tokens.extend(bpe_token for bpe_token in tokenizer.bpe(token).split(" "))
    return tokens


def convert_tokens_to_id(tokenizer: SimpleTokenizer, words: str, max_words: int = 32) -> np.ndarray:
    """
    Convert tokens to token ID.
    Args:
        tokenizer (`SimpleTokenizer`):
            SimpleTokenizer instance.
        words (`str`):
            Raw text words.
        max_words (`int`):
            Max mord length, if not enough, the output ID is 0.

    Returns:
        Ndarray of ID list.
    """
    special_token = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                     "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    pairs_text = np.zeros((1, max_words), dtype=np.int32)

    words = tokenize(words)
    words = [special_token["CLS_TOKEN"]] + words
    total_length_with_cls = max_words - 1
    if len(words) > total_length_with_cls:
        words = words[:total_length_with_cls]
    words = words + [special_token["SEP_TOKEN"]]

    input_ids = [tokenizer.encoder[bpe_token] for bpe_token in words]
    while len(input_ids) < max_words:
        input_ids.append(0)
    assert len(input_ids) == max_words

    pairs_text[0] = np.array(input_ids)
    return pairs_text
