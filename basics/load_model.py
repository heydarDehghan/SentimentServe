from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# # Load the pre-trained model and tokenizer
# model_name = 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis'
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# # Save the model and tokenizer locally
# model.save_pretrained('model/saved_model')
# tokenizer.save_pretrained('tokenizer/saved_model')

# Load the model and tokenizer from the local directory
loaded_model = AutoModelForSequenceClassification.from_pretrained('model/saved_model')
loaded_tokenizer = AutoTokenizer.from_pretrained('tokenizer/saved_model')

# Create a new sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis', model=loaded_model, tokenizer=loaded_tokenizer)

# Example usage
text = "I absolutely loved this movie! It was fantastic."
result = sentiment_pipeline(text)
print(result)
