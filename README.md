# Multilingual Proverbs: Creating a Dataset and Analyzing Irony through Sentiment Analysis

## Abstract

Figurative language, such as irony, enriches human communication by conveying meanings beyond literal expressions. However, detecting irony poses significant
challenges for Natural Language Processing (NLP) due to its subtlety and cultural
specificity. This study investigates the presence of irony in proverbs across Greek,
German, and Italian languages by analyzing the sentiment differences between their
literal English translations and intended meanings. We constructed a custom multilingual dataset of proverbs, performed preprocessing, and applied various sentiment
analysis models, including VADER, DistilBERT, RoBERTa, and a combined approach
utilizing FLAN-T5 for text simplification and DistilBERT for sentiment analysis. Our
experiments revealed that while advanced transformer-based models achieved an overall accuracy of approximately 60% in detecting irony, neither the combined approach
nor the direct irony classification method did not significantly enhance performance,
underscoring the inherent complexity of irony detection. These findings highlight
the limitations of current sentiment analysis techniques in capturing the nuances of
figurative language and suggest the need for models capable of deeper semantic understanding. Future work involves incorporating more sophisticated models and exploring irony detection across different languages and cultures.

## Introduction

In this project, we explored the linguistic variations and sentiment conveyed through proverbs and sayings in our native languages—Greek, German, and Italian. Motivated by the frequent differences we observed during everyday conversations, we aimed to investigate whether irony is present in proverbs by leveraging natural language processing (NLP) techniques.

Irony detection has been studied in various contexts, including social media analysis and sentiment analysis. Early approaches relied on linguistic features and hand-crafted rules [1].

Recent studies have leveraged transformer-based models for irony detection. For instance, researchers have fine-tuned pre-trained language models like BERT and RoBERTa to capture the complex contextual and semantic nuances associated with irony [2].

Metaphor interpretation has also been a focus of research. Liu et al. [3] tested the ability of language models to interpret figurative language, particularly metaphors, and found that fine-tuning and prompt engineering can enhance performance. Guzman Piedrahita et al. [4] extended this work by exploring the gap between sentence probabilities and decoded outputs in metaphor interpretation.

Our work builds upon these studies by applying sentiment analysis to detect irony in proverbs, which are inherently figurative and culturally specific. By analyzing sentiment differences, we aim to identify ironic expressions that traditional models might overlook.

Our work builds upon these studies by applying sentiment analysis to detect irony in proverbs, which are inherently figurative and culturally specific. By analyzing sentiment differences, we aim to identify ironic expressions that traditional models might overlook.
The problem we addressed was how to detect irony in proverbs by analyzing the sentiment difference between their literal English translations and their intended meanings. This project lies in the area of sentiment analysis, a major subfield of NLP, with a focus on sentiment comparison to identify linguistic features such as irony. Our approach involved creating a dataset, performing sentiment analysis using transformer-based models, and validating our method against manually labeled data.


Drawing parallels between metaphor and irony detection, we aim to contribute to the broader understanding of figurative language processing in NLP.

## Dataset

For this project, we created our own dataset containing proverbs and sayings in Greek, German, and Italian, as these are the native languages of our team members. The dataset was constructed by gathering proverbs and their translations from various online sources, including websites and generative AI tools. We designed the dataset specifically to address our project’s needs, ensuring that it contained the necessary linguistic and cultural elements relevant to our analysis of irony. The dataset includes the following columns:

- **Proverb**: The original proverb in its native language (Greek, German, or Italian).
- **Language**: The language in which the proverb is written.
- **Literal English Translation**: A direct translation of the proverb into English.
- **Meaning**: The intended meaning of the proverb, also translated into English.
- **Irony**: A binary label (Yes/No) indicating whether the proverb is ironic or not.

An example from the dataset:

| **Language** | **Proverb**             | **Literal English Translation** | **Meaning**                | **Irony** |
|--------------|-------------------------|---------------------------------|----------------------------|-----------|
| German       | Das ist nicht mein Bier | This is not my beer             | This is not my problem     | Yes       |

We chose to construct the dataset ourselves because no existing dataset suited our specific combination of languages and the focus on irony in proverbs. Additionally, creating our own dataset allowed us to tailor the content to meet the needs of our analysis, ensuring a balance of languages and diverse examples.

Alternative options could have included using an existing dataset, though we were unable to find one that matched our languages and requirements. Another option would have been to automate the dataset creation through web scraping, but we opted for manual curation to ensure quality and relevance in the selected proverbs.

## Preprocessing

We applied different auxiliary NLP tasks to preprocess the data in both the "Literal English Translation" and "Meaning" columns. These tasks were essential to prepare the data for accurate sentiment analysis and to reduce any noise or inconsistencies.

The preprocessing steps we used are:

1. **Lowercasing**: All text was converted to lowercase to ensure uniformity and avoid discrepancies caused by capitalization, which could influence the sentiment analysis.
2. **Tokenization**: The text was tokenized into individual words using NLTK's `word_tokenize` function. Tokenization breaks down the text into smaller units (tokens) that can be analyzed more effectively in the sentiment analysis process.
3. **Stopword Removal**: We removed common stopwords (such as "the," "and," etc.) using NLTK's list of stopwords since those typically don't carry sentiment.
4. **Punctuation Removal**: Punctuation was removed to avoid interference with the sentiment analysis, as punctuation is generally irrelevant when calculating sentiment.

This preprocessing was implemented using Python’s NLTK library. The cleaned and preprocessed text allowed for more consistent and meaningful sentiment analysis results, helping us to improve overall accuracy.

## Model Training

### Training Process

We initiated our sentiment analysis using pre-trained models from Hugging Face's Transformers library. The steps involved were:

1. **Loading the Models**: We loaded the pre-trained models and their respective tokenizers.
2. **Preprocessing the Data**: The dataset was preprocessed as described in the previous section.
3. **Sentiment Analysis Pipeline**: We utilized sentiment analysis pipelines to evaluate both the "Literal English Translation" and the "Meaning" of each proverb.
4. **Sentiment Difference Calculation**: By comparing the sentiment scores or the sentiment labels of the literal translation and the intended meaning, we aimed to identify discrepancies indicative of irony.
5. **Irony Classification**: Based on a predefined threshold or label disparities, we classified proverbs as "Ironic" or "Not Ironic."

### Models Used

In order to improve the performance metrics and evaluate the suitability of different models, we experimented with different models and combined approaches. These models were chosen for their superior language understanding and ability to capture nuanced sentiments, which are crucial for detecting irony—a sophisticated form of figurative language.

- **VADER Analysis Model**: Our initial approach employed the VADER sentiment analysis model from the NLTK library. VADER is a rule-based model particularly effective for social media texts but applicable to other domains. However, since this approach did not perform significantly better than chance level, we decided not to further consider it in the following evaluations.

- **DistilBERT-based Sentiment Analysis**: DistilBERT is a streamlined, smaller version of the BERT model that maintains much of BERT’s ability to understand contextual nuances in text. It uses a transformer architecture optimized for faster, more efficient sentiment analysis, making it well-suited for applications requiring a balance between accuracy and computational efficiency.

- **RoBERTa-based Sentiment Analysis**: RoBERTa (Robustly Optimized BERT Approach) is an enhancement of BERT, trained with improved methods like dynamic masking and larger batch sizes [5]. These optimizations allow RoBERTa to better capture subtle contextual information, making it particularly effective in understanding the layered sentiments often found in ironic statements.

- **Twitter-RoBERTa Irony**: This is a RoBERTa-base model trained on approximately 58 million tweets and fine-tuned for irony detection with the TweetEval benchmark [6]. Its output is a classification of 'ironic' and 'non-ironic' statements. The use of this method does not require any of our methodology steps and can be used directly on the Literal Translated Proverbs.

- **Combined Approach with DistilBERT and FLAN-T5 for Simplification**: To enhance sentiment analysis further, we implemented a combined approach that includes a text simplification step before sentiment analysis, inspired by a project from last year's class by David Guzman Piedrahita, Rajiv Bains, and Lucas Krauter that used similar methods to improve sentiment accuracy. We used `distilbert-base-uncased-finetuned-sst-2-english` for sentiment analysis alongside `google/flan-t5-large` for text simplification. The FLAN-T5 model simplifies the `Literal English Translation` column, potentially reducing linguistic complexity and making sentiment patterns clearer. After simplification, sentiment analysis is applied to both original and simplified text versions.

## Evaluation

### Metrics

We measured the following metrics to evaluate the performance of our models:

- **Precision**: The proportion of correctly predicted ironic cases out of all cases predicted as ironic. High precision means the model is good at avoiding false positives.

- **Recall**: The proportion of correctly predicted ironic cases out of all actual ironic cases. High recall indicates the model is good at capturing most ironic instances.

- **Specificity**: The proportion of correctly predicted non-ironic cases out of all actual non-ironic cases. High specificity means the model is good at correctly identifying non-ironic instances and avoiding false positives among non-ironic cases.

- **Accuracy**: The proportion of correct predictions (ironic or not ironic) out of all predictions made. It gives a general measure of model performance.

### Results of Advanced Transformer-Based Models

The results, summarized in the table below, indicate that while advanced models like DistilBERT and RoBERTa show some improvement over chance level, the overall accuracy remains around 60%. The combined approach did not significantly enhance performance, suggesting the complexity of irony detection.

Additionally, all the models which predict irony based on the sentiment comparison of the Literal English Translation and the Meaning perform significantly better than the model which is specifically trained to detect irony (Irony RoBERTa) and uses the Literal English Translation.

#### Performance Metrics of Different Models

| **Model**           | **Precision** | **Recall** | **Specificity** | **Accuracy** |
|---------------------|---------------|------------|-----------------|--------------|
| DistilBERT          | 0.60          | 0.69       | 0.42            | 0.57         |
| RoBERTa             | 0.62          | 0.61       | 0.52            | 0.57         |
| Irony RoBERTa       | 0.46          | 0.22       | 0.67            | 0.42         |
| Combined Approach   | 0.59          | 0.83       | 0.24            | 0.57         |

## Current Challenges

One of the main challenges in this project is the inherent complexity of detecting irony. Irony is difficult to define and often highly subjective, varying depending on cultural and linguistic contexts. What may be perceived as ironic in one language or culture might not be seen the same way in another, making it a challenging concept to quantify.

Additionally, most sentiment analysis tools primarily focus on the sentiment of individual words or phrases. These tools do not always account for the deeper semantic context required to detect irony. Irony often relies on subtle shifts in meaning or contradictions between what is said and what is meant, which can be challenging for sentiment analysis models that work on word-level polarity. Without the ability to "read between the lines" or understand the broader context of the proverb, these tools may miss the nuanced sentiment shifts that signify irony.

## Future Work

There are several potential avenues for extending this work in the future, mainly addressing the challenges mentioned in the previous section. The main areas for future work we identified are:

- **Incorporate more sophisticated models** capable of detecting semantics and the broader context. Leveraging deep learning models, such as transformer-based architectures like BERT or GPT, could allow for a better understanding of the nuanced meanings between the literal and intended interpretations of proverbs. This would improve the accuracy of irony detection by considering semantic context, rather than using only word-level sentiment.

- **Identify more appropriate quantifiers for irony** to enhance the classification process. Currently, we use manually added labels based on our own knowledge of the languages; however, this approach is highly subjective and might introduce bias. Future research could explore new metrics to accurately define irony.

- **Use this approach to compare the expression of irony across different languages and cultures**. By expanding the dataset to include proverbs from additional languages and applying more refined irony detection methods, future work could explore how irony is conveyed differently across linguistic and cultural boundaries, providing deeper insights into cross-cultural communication.

- **Treat the threshold of allowed sentiment variance as a hyperparameter**. This threshold defines how much margin between the sentiment of the literal statement and the sentiment of the meaning can exist without defining a proverb as ironic. This variable is set a priori and has a big influence on the results. Therefore, an NLP engineer should be allowed to easily change and tune this parameter via grid search or with other methods of hyperparameter tuning to achieve the best results.

Moreover, drawing inspiration from Guzman Piedrahita et al.'s work [4], which highlighted the gap between sentence probabilities and human-perceived correctness, future work could explore similar gaps in irony detection. Implementing human evaluations alongside automated metrics can provide a more holistic assessment of the models' performance, ensuring that the detected irony aligns with human interpretations.

## References

1. **Reyes, A., Rosso, P., & Buscaldi, D. (2013)**. *From humor recognition to irony detection: The figurative language of social media*. Data & Knowledge Engineering, 74, 1-12.

2. **Cignarella, V. P., Van Hee, A., Bosco, C., Patti, V., & Daelemans, W. (2020)**. *Overview of the EVALITA 2020 Task on Irony Detection in Italian Tweets (IronITA)*. In Proceedings of the 7th Evaluation Campaign of Natural Language Processing and Speech Tools for Italian (EVALITA 2020).

3. **Liu, E., Cui, C., Zheng, K., & Neubig, G. (2022)**. *Testing the Ability of Language Models to Interpret Figurative Language*. arXiv preprint arXiv:2206.05513.

4. **Guzman Piedrahita, D., Bains, R., & Krauter, L. (2024)**. *Metaphors Unveiled: Exploring Language Models for Figurative Text and the Gap Between Sentence Probabilities and Decoded Text*.

5. **Camacho-Collados, J., Pilehvar, M. T., & Navigli, R. (2022)**. *TweetNLP: Cutting-Edge Natural Language Processing for Social Media*. arXiv preprint arXiv:2201.00000.

6. **Barbieri, F., Camacho-Collados, J., Espinosa-Anke, L., & Neves, L. (2020)**. *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification*. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP).

## How to Run the Code

### Open the Notebook:

Navigate to the desired notebook within the models/ directory (e.g., RoBERTa_Irony.ipynb, RoBERTa_sentiment.ipynb, distilbert_sentiment.ipynb, etc.).
Open the notebook in Google Colab or any environment that supports Jupyter notebooks.
Install Dependencies:

Each notebook contains a command to install necessary dependencies from the requirements.txt file, so there’s no need for separate environment setup.

Execute the first code cell in the notebook, which includes %pip install -r ../requirements.txt. This will automatically install all dependencies required for the notebook.
### File Setup:

Ensure that the proverbs.csv dataset is located in the data/ directory, which is referenced within the code cells.



## Repository Structure

```
multilingual-proverbs-irony/
├── data/
│   └── proverbs.csv
├── models/
│   └── RoBERTa_Irony.ipynb
│   └── RoBERTa_sentiment.ipynb
│   └── distilbert_sentiment.ipynb
│   └── combined_approach.ipynb
├── README.md
├── requirements.txt

```

- **data/**: Contains the dataset.
- **models/**: Directory for saving trained models (if applicable).
- **README.md**: Detailed project description.
- **requirements.txt**: Package requirements.


