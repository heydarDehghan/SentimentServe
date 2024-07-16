# Sentiment Analysis API using BERT

This repository contains a Flask-based API for sentiment analysis using a pre-trained BERT model. The API provides endpoints for analyzing single sentences, batches of sentences, and evaluating the model's performance.

## Features

- Analyze sentiment for a single sentence.
- Analyze sentiment for an array of sentences.
- Evaluate the model using common metrics.
- Configuration file for easy parameter management.

## Installation

1. **Install the required packages:**
    ```bash
    pip install flask transformers torch
    ```

2. **Prepare the configuration file:**

    Edit the `config.json` file to include the paths to your BERT model and tokenizer, and set other parameters as needed.
    ```json
    {
        "model_path": "path/to/bert/model",
        "tokenizer_path": "path/to/bert/tokenizer",
        "max_length": 128,
        "evaluation_metrics": ["accuracy", "precision", "recall"]
    }
    ```

## Usage

1. **Run the API:**
    ```bash
    python app.py
    ```

2. **API Endpoints:**

    - **Analyze a single sentence:**
        - Endpoint: `/sentiment`
        - Method: `POST`
        - Request body:
            ```json
            {
                "sentence": "Your sentence here"
            }
            ```
        - Response:
            ```json
            {
                "sentence": "Your sentence here",
                "sentiment_score": 0.95,
                "sentiment_label": 1
            }
            ```

    - **Analyze an array of sentences:**
        - Endpoint: `/batch_sentiment`
        - Method: `POST`
        - Request body:
            ```json
            {
                "sentences": ["Sentence one", "Sentence two"]
            }
            ```
        - Response:
            ```json
            [
                {
                    "sentence": "Sentence one",
                    "sentiment_score": 0.95,
                    "sentiment_label": 1
                },
                {
                    "sentence": "Sentence two",
                    "sentiment_score": 0.87,
                    "sentiment_label": 0
                }
            ]
            ```

    - **Evaluate the model:**
        - Endpoint: `/evaluate`
        - Method: `GET`
        - Response:
            ```json
            {
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0
            }
            ```
        (Note: Replace the dummy evaluation function with actual evaluation logic)

## Configuration

The `config.json` file contains the following parameters:

- `model_path`: Path to the BERT model.
- `tokenizer_path`: Path to the BERT tokenizer.
- `max_length`: Maximum token length for input sentences.
- `evaluation_metrics`: List of metrics to use for model evaluation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## License

This project is licensed under the MIT License.
