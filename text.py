import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Load the customer comments data
comments = pd.read_csv("comments.csv")
comments = comments['comments'].tolist()

# Load the BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Encode the customer comments
encoded_comments = [tokenizer.encode(comment, return_tensors='pt', max_length=512, padding=True, truncation=True) for comment in comments]

# Use BERT to get the topic representation for each comment
topic_representations = []
for comment in encoded_comments:
    with torch.no_grad():
        last_hidden_state = model(comment)[0].last_hidden_state
        avg_pooled = torch.mean(last_hidden_state, dim=1)
        topic_representations.append(avg_pooled.numpy().flatten())

# Use the topic representations to perform topic modeling
from sklearn.decomposition import NMF
n_components = 5
model = NMF(n_components=n_components, init='random', random_state=0)
W = model.fit_transform(topic_representations)

# Print the topics
for topic_index in range(n_components):
    topic = model.components_[topic_index]
    topic_words = ", ".join([tokenizer.convert_ids_to_tokens(i) for i in topic.argsort()[:-6:-1]])
    print("Topic", topic_index, ":", topic_words)

# Use BERT to perform sentiment analysis on the customer comments
sentiments = []
for comment in encoded_comments:
    with torch.no_grad():
        logits = model(comment)[0][:,0]
        sentiment = torch.sigmoid(logits).item()
    sentiments.append(sentiment)

# Add the sentiment scores to the customer comments data
comments['sentiment'] = sentiments
