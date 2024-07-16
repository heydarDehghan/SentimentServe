# app.py
from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, pipeline, \
    AutoModelForSequenceClassification
import pandas as pd
import json

app = Flask(__name__)

# Load configuration
with open('static/config.json') as config_file:
    config = json.load(config_file)

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(config['model_path'])
tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])


def analyze_sentiment(sentence):
    # inputs = tokenizer(sentence, return_tensors='pt', max_length=config['max_length'], truncation=True,
    #                    padding='max_length')
    # outputs = model(**inputs)
    # probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # sentiment_score = torch.max(probs, dim=1).values.item()
    # sentiment_label = torch.argmax(probs, dim=1).item()

    # Create a new sentiment analysis pipeline
    sentiment_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    result = sentiment_pipeline(sentence)

    return result
    # return sentiment_score, sentiment_label


@app.route('/sentiment', methods=['POST'])
def get_sentiment():
    data = request.json
    sentence = data['sentence']
    sentiment_score = analyze_sentiment(sentence)
    return sentiment_score


@app.route('/batch_sentiment', methods=['POST'])
def get_batch_sentiment():
    data = request.json
    sentences = data['sentences']
    results = []
    for sentence in sentences:
        sentiment_score, sentiment_label = analyze_sentiment(sentence)
        results.append({'sentence': sentence, 'sentiment_score': sentiment_score, 'sentiment_label': sentiment_label})
    return jsonify(results)


def accuracy_score(true_labels, pred_labels):
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    return correct / len(true_labels)


def precision_score(true_labels, pred_labels):
    true_positive = sum((t == 1 and p == 1) for t, p in zip(true_labels, pred_labels))
    false_positive = sum((t == 0 and p == 1) for t, p in zip(true_labels, pred_labels))
    if true_positive + false_positive == 0:
        return 0.0
    return true_positive / (true_positive + false_positive)


def recall_score(true_labels, pred_labels):
    true_positive = sum((t == 1 and p == 1) for t, p in zip(true_labels, pred_labels))
    false_negative = sum((t == 1 and p == 0) for t, p in zip(true_labels, pred_labels))
    if true_positive + false_negative == 0:
        return 0.0
    return true_positive / (true_positive + false_negative)


def evaluate_model():
    # Load dataset
    dataset = pd.read_csv(config['dataset_path'])
    texts = dataset['text'].tolist()
    true_labels = dataset['label'].tolist()

    # Get predictions
    pred_labels = []
    for text in texts:
        _, sentiment_label = analyze_sentiment(text)
        pred_labels.append(sentiment_label)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }


@app.route('/evaluate', methods=['GET'])
def evaluate():
    evaluation_results = evaluate_model()
    return jsonify(evaluation_results)


if __name__ == '__main__':
    app.run(debug=True)
