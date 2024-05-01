from typing import Literal

import spacy
import torch
import pandas as pd
import spacy_cleaner
import spacy_fastlang
from transformers import pipeline
from spacy.tokens.doc import Doc
from spacy_cleaner.processing import mutators, removers, replacers


def language_detection(
    df: pd.DataFrame,
    text_col: str,
    model_type: Literal["roberta", "fasttext"] = "fasttext",
) -> pd.DataFrame:
    """
    https://huggingface.co/papluca/xlm-roberta-base-language-detection
    https://spacy.io/universe/project/spacy_fastlang
    """
    dfc = df.copy()

    if model_type == "fasttext":

        def _get_language(token: Doc) -> str:
            return token._.language

        def _get_language_score(token: Doc) -> str:
            return token._.language_score

        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("language_detector")
        res = df[text_col].astype(str).apply(nlp)
        dfc[text_col + "_language"] = res.apply(_get_language)
        dfc[text_col + "_language_score"] = res.apply(_get_language_score)
    elif model_type == "roberta":

        def _get_language(token: Doc) -> str:
            return token._.language

        def _get_language_score(token: Doc) -> str:
            return token._.language_score

        pipe = pipeline(
            "text-classification", model="papluca/xlm-roberta-base-language-detection"
        )
        res = pipe(dfc["CustomerFeedback"].astype(str).to_list())
        dfc[text_col + "_language"] = pipe(dfc[text_col].astype(str))
    else:
        raise NotImplementedError
    return dfc


def text_cleaning(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Returns dataframe with preprocessed/cleaned text column.
    Spacy-cleaner that uses spacy functionalities.
    https://spacy.io/universe/project/spacy-cleaner

    Note:
        The spacy-cleaner library does not do much.
        Future task - develop own cleaner

    Args:
        df (pd.DataFrame): input dataset
        text_col: column name with text
    """
    dfc = df.copy()
    model = spacy.load("en_core_web_sm")
    cleaner = spacy_cleaner.Cleaner(
        model,
        removers.remove_stopword_token,
        replacers.replace_punctuation_token,
        mutators.mutate_lemma_token,
    )
    dfc[text_col] = cleaner.clean(dfc[text_col].astype(str))

    return dfc


def sentiment_analysis(
    df: pd.DataFrame, text_col: str, sentiment_depth: Literal[3, 5] = 3
) -> pd.DataFrame:
    """
    Returns dataframe with new column that analysis sentintent in the text_col.

    initial idea:
    Inspired by https://www.nature.com/articles/s41598-024-60210-7 I also used
    the 3 most popular models with voting:
    1. cardiffnlp/twitter-roberta-base-sentiment-latest
    2. nlptown/bert-base-multilingual-uncased-sentiment
    3. mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
    4. lxyuan/distilbert-base-multilingual-cased-sentiments-student
    5. finiteautomata/bertweet-base-sentiment-analysis

    issues:
    1. 3(pos, neutral, neg) vs 5(1-5 stars) sentiments; models 1,3 vs 2
    2. maximum sequence length models 4-5

    final idea:
    Either 3 sentimnets(model 1) or 5 stars (model 2)

    Args:
        df (pd.DataFrame): input dataset
        text_col (str): column name with text
        sentiment_depth ([3, 5]): depth of sentiment analysis
    """
    dfc = df.copy()
    if sentiment_depth == 3:
        pipe = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    elif sentiment_depth == 5:
        pipe = pipeline(
            "text-classification",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    else:
        raise NotImplementedError
    res = pipe(dfc[text_col].astype(str).to_list())
    res = pd.DataFrame(res).rename(
        columns={
            "label": text_col + "_sentiment",
            "score": text_col + "_sentiment_score",
        }
    )
    return pd.concat([dfc, res], axis=1)
