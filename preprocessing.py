import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess(df: pd.DataFrame):

    #get first 100 elements
    df = df.head(300)

    df.dropna(inplace=True)
    df["processed"] = df["text"].apply(process_text)

    return df


def process_text(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    return " ".join(filtered_tokens)
    