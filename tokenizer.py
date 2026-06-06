import re

import jieba


def tokenize_for_search(text: str) -> str:
    words = jieba.cut(text)
    cleaned_words: list[str] = []

    for word in words:
        word = word.strip().lower()
        if not word:
            continue
        if re.fullmatch(r"\W+", word):
            continue
        cleaned_words.append(word)

    return " ".join(cleaned_words)
