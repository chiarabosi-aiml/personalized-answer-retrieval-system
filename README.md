# Personalized Answer Retrieval — Information Retrieval and Recommender Systems

## Overview

This project explores a **personalized information retrieval system** that ranks relevant answers for natural-language queries using the **SE-PQA** dataset. We compare lexical and neural retrieval approaches, then investigate whether query expansion using contextual tags improves the ranking.

## Approach

1. **Text preprocessing:** clean query and answer text, remove noise and explore tokenization, stopword removal and lemmatization.
2. **TF-IDF baseline:** vectorize text, compute cosine similarities and retrieve top-ranked answers.
3. **Neural retrieval:** index documents with PyTerrier, compare TF-IDF, bi-encoder, cross-encoder and hybrid ranking pipelines.
4. **Model evaluation:** compare retrieval using MAP, NDCG and Precision@10 on held-out queries.
5. **Context-aware query expansion:** use query tags and the pretrained `microsoft/Phi-3-mini-4k-instruct` model to generate expanded queries and evaluate the resulting retrieval.

## Findings

In the documented experiments, the **bi-encoder pipeline performed best** among the compared approaches. The tested query expansion reduced the reported retrieval metrics rather than improving them, suggesting that adding context can introduce irrelevant terms or shift rankings.

The study used reduced samples of **50 and 250 queries** because of computational constraints. These findings should not be interpreted as an evaluation over the full dataset. See the [original report](docs/PROJECT_REPORT.pdf) for methodology, figures and limitations.

## Repository contents

- [`PROJECT.ipynb`](PROJECT.ipynb) — original notebook, including its saved outputs.
- [`docs/PROJECT_REPORT.pdf`](docs/PROJECT_REPORT.pdf) — original academic report.
- [`requirements.txt`](requirements.txt) — indicative dependencies; **not a verified reproducible environment**.

The dataset and pretrained model weights are not distributed in this repository.

## Running the notebook

The notebook was developed in a Google Colab-style environment and requires internet access, model downloads and significant compute resources (the Phi-3 query-expansion section is configured for CUDA). The original cells download data using `gdown`, install some dependencies, build a PyTerrier index and load Hugging Face models.

Install the indicative dependencies with `pip install -r requirements.txt`, open `PROJECT.ipynb` and follow its cells. **Execution has not been revalidated in a fresh environment**; version changes and unavailable external download links may require environment adjustments. Do not assume rerunning will reproduce all saved results.

## Authors and contributions

Joint coursework project by **Chiara Bosi**, **Leonardo Castelli** and **Francesca Chite**. 

## Publication and attribution

The notebook and report are shared for academic portfolio purposes. Dataset files are omitted; check the original data source's terms before downloading or redistributing.
