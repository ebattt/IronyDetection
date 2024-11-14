# Multilingual Proverbs: Creating a Dataset and Analyzing Irony through Sentiment Analysis

## Abstract

Figurative language, such as irony, enriches human communication by conveying meanings beyond literal expressions. However, detecting irony poses significant challenges for Natural Language Processing (NLP) due to its subtlety and cultural specificity. This study investigates the presence of irony in proverbs across Greek, German, and Italian languages by analyzing the sentiment differences between their literal English translations and intended meanings.

We constructed a custom multilingual dataset of proverbs, performed preprocessing, and applied various sentiment analysis models, including VADER, DistilBERT, RoBERTa, and a combined approach utilizing FLAN-T5 for text simplification and DistilBERT for sentiment analysis. Our experiments revealed that while advanced transformer-based models achieved an overall accuracy of approximately 60% in detecting irony, the combined approach did not significantly enhance performance, underscoring the inherent complexity of irony detection.

These findings highlight the limitations of current sentiment analysis techniques in capturing the nuances of figurative language and suggest the need for models capable of deeper semantic understanding. Future work involves incorporating more sophisticated models and exploring irony detection across different languages and cultures.

## Introduction

In this project, we explored the linguistic variations and sentiment conveyed through proverbs and sayings in our native languages—Greek, German, and Italian. Motivated by the frequent differences we observed during everyday conversations, we aimed to investigate whether irony is present in proverbs by leveraging Natural Language Processing (NLP) techniques.

Irony detection has been studied in various contexts, including social media analysis and sentiment analysis. Early approaches relied on linguistic features and hand-crafted rules [1]. Recent studies have leveraged transformer-based models for irony detection. For instance, researchers have fine-tuned pre-trained language models like BERT and RoBERTa to capture the complex contextual and semantic nuances associated with irony [2].

Metaphor interpretation has also been a focus of research. Liu et al. [3] tested the ability of language models to interpret figurative language, particularly metaphors, and found that fine-tuning and prompt engineering can enhance performance. Guzman Piedrahita et al. [4] extended this work by exploring the gap between sentence probabilities and decoded outputs in metaphor interpretation.

Our work builds upon these studies by applying sentiment analysis to detect irony in proverbs, which are inherently figurative and culturally specific. By analyzing sentiment differences, we aim to identify ironic expressions that traditional models might overlook.

## Dataset

We created our own dataset containing proverbs and sayings in Greek, German, and Italian, as these are the native languages of our team members. The dataset includes the following columns:

- **Proverb**: The original proverb in its native language (Greek, German, or Italian).
- **Language**: The language in which the proverb is written.
- **Literal English Translation**: A direct translation of the proverb into English.
- **Meaning**: The intended meaning of the proverb, also translated into English.
- **Irony (Yes/No)**: A binary label indicating whether the proverb is ironic or not.

An example from the dataset:

| **Language** | **Proverb**              | **Literal English Translation** | **Meaning**                          | **Irony** |
|--------------|--------------------------|---------------------------------|--------------------------------------|-----------|
| German       | Das ist nicht mein Bier  | This is not my beer             | This is not my problem               | Yes       |

The dataset is located in the `data/` directory as `proverbs.csv`.

## Preprocessing

We applied several preprocessing steps to the "Literal English Translation" and "Meaning" columns:

1. **Lowercasing**: Converted all text to lowercase.
2. **Tokenization**: Split text into individual words using NLTK's `word_tokenize` function.
3. **Stopword Removal**: Removed common English stopwords using NLTK's stopword list.
4. **Punctuation Removal**: Removed punctuation marks to prevent interference with sentiment analysis.

These steps help reduce noise and standardize the text for better sentiment analysis performance.

## Models Used

### VADER Sentiment Analysis Model

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based sentiment analysis tool specifically attuned to sentiments expressed in social media. It's part of the NLTK library and doesn't require additional training.

### DistilBERT-based Sentiment Analysis

DistilBERT is a smaller, faster, and lighter version of BERT, retaining much of its language understanding capabilities. It's suitable for tasks requiring efficiency without significant loss in performance.

### RoBERTa-based Sentiment Analysis

RoBERTa (Robustly Optimized BERT Pretraining Approach) improves upon BERT with training optimizations, making it more effective at capturing subtle contextual information.

### T5-small for Text Simplification and Sentiment

T5-small is a smaller version of the Text-to-Text Transfer Transformer (T5) model. We used it for text simplification to reduce linguistic complexity before sentiment analysis.

### Combined Approach with DistilBERT and FLAN-T5 for Simplification

Inspired by previous work, we combined FLAN-T5 for text simplification with DistilBERT for sentiment analysis to see if simplification improves irony detection.

## Training Process

1. **Loading Models**: Loaded pre-trained models and tokenizers from Hugging Face's Transformers library.
2. **Preprocessing**: Applied the preprocessing steps to the dataset.
3. **Sentiment Analysis**: Performed sentiment analysis on both the "Literal English Translation" and "Meaning" columns.
4. **Sentiment Difference Calculation**: Calculated the difference between sentiment scores to detect irony.
5. **Irony Classification**: Classified proverbs as "Ironic" or "Not Ironic" based on sentiment differences.

## Evaluation

### Metrics

- **Accuracy**: Overall correctness of the model's predictions.
- **Precision**: Correctly predicted ironic proverbs out of all predicted ironic.
- **Recall (Sensitivity)**: Correctly predicted ironic proverbs out of all actual ironic proverbs.
- **F1 Score**: Harmonic mean of precision and recall.
- **Specificity**: Correctly predicted non-ironic proverbs out of all actual non-ironic proverbs.

### Results

#### Performance Metrics of Different Models

| **Model**          | **Precision** | **Recall** | **Specificity** | **Accuracy** |
|--------------------|---------------|------------|-----------------|--------------|
| VADER              | 0.61          | 0.81       | 0.30            | 0.60         |
| DistilBERT         | 0.62          | 0.81       | 0.31            | 0.60         |
| RoBERTa            | 0.62          | 0.81       | 0.31            | 0.60         |
| Combined Approach  | 0.61          | 0.83       | 0.26            | 0.59         |

#### Per-Language Performance (RoBERTa Model)

| **Language** | **Precision** | **Recall** | **Specificity** | **Accuracy** |
|--------------|---------------|------------|-----------------|--------------|
| Greek        | 0.40          | 0.13       | 0.80            | 0.47         |
| German       | 0.69          | 0.43       | 0.73            | 0.56         |
| Italian      | 0.89          | 0.44       | 0.92            | 0.63         |

## Challenges

- **Subjectivity of Irony**: Irony is culturally specific and subjective, making it difficult to detect.
- **Contextual Understanding**: Sentiment analysis models may not capture the deeper semantic context required for irony detection.
- **Dataset Size**: Limited dataset size may affect the generalizability of the models.

## Future Work

- **Advanced Models**: Incorporate models like BERT or GPT with fine-tuning for irony detection.
- **Improved Metrics**: Develop more objective quantifiers for irony.
- **Cross-Cultural Analysis**: Expand the dataset to include more languages and cultures for a broader analysis.

## How to Run the Code

### Prerequisites

- **Python 3.7** or higher
- Install required packages:

```bash
pip install -r requirements.txt
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/IronyDetection.git
cd IronyDetection 
```

### Dataset

Ensure the `proverbs.csv` dataset is located in the `data/` directory.

### Running the Script

```bash
python scripts/irony_detection.py
```

This script will:

- Load and preprocess the dataset.
- Perform sentiment analysis using the selected models.
- Calculate sentiment differences.
- Classify proverbs as ironic or not ironic.
- Evaluate model performance and display metrics.

### Output

- Evaluation metrics will be displayed in the console.
- Results and figures will be saved in the `output/` directory.

## Dependencies and Requirements

- **pandas**
- **numpy**
- **nltk**
- **torch**
- **transformers**
- **scikit-learn**
- **matplotlib**

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Repository Structure

```
multilingual-proverbs-irony/
├── data/
│   └── proverbs.csv
├── scripts/
│   └── irony_detection.py
├── output/
│   ├── figures/
│   └── results/
├── models/
├── README.md
├── requirements.txt
├── LICENSE
```

- **data/**: Contains the dataset.
- **scripts/**: Python scripts for data processing and analysis.
- **output/**: Contains output files like evaluation results and figures.
- **models/**: Directory for saving trained models (if applicable).
- **README.md**: Detailed project description.
- **requirements.txt**: Python package requirements.
- **LICENSE**: Project license information.

## References

1. **Reyes, A., Rosso, P., & Buscaldi, D. (2013)**. *From humor recognition to irony detection: The figurative language of social media*. Data & Knowledge Engineering, 74, 1-12.

2. **Cignarella, V. P., Van Hee, A., Bosco, C., Patti, V., & Daelemans, W. (2020)**. *Overview of the EVALITA 2020 Task on Irony Detection in Italian Tweets (IronITA)*. In Proceedings of the 7th Evaluation Campaign of Natural Language Processing and Speech Tools for Italian (EVALITA 2020).

3. **Liu, E., Cui, C., Zheng, K., & Neubig, G. (2022)**. *Testing the Ability of Language Models to Interpret Figurative Language*. arXiv preprint arXiv:2206.05513.

4. **Guzman Piedrahita, D., Bains, R., & Krauter, L. (2024)**. *Metaphors Unveiled: Exploring Language Models for Figurative Text and the Gap Between Sentence Probabilities and Decoded Text*.

---

If you have any questions or need further assistance, please feel free to contact us.
