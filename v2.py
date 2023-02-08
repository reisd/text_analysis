import pandas as pd
import torch
import transformers
from sklearn.decomposition import NMF

# Load the BERT model
model = transformers.BertModel.from_pretrained("bert-base-uncased")
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

# Read the customer comments from the CSV file as a DataFrame
df = pd.read_csv("comments.csv")
comments = df["Comment"].tolist()

# Encode the comments as input for the BERT model
encoded_comments = [tokenizer.encode(comment, add_special_tokens=True) for comment in comments]
padded_comments = torch.nn.utils.rnn.pad_sequence(encoded_comments, batch_first=True)

# Get the topic representations for each comment
with torch.no_grad():
    _, last_hidden_state = model(padded_comments)
topic_representations = last_hidden_state[0].mean(dim=1)

# Perform sentiment analysis on the comments
sentiments = []
for i, comment in enumerate(comments):
    sentiment = 0
    for token in comment:
        sentiment += topic_representations[i, tokenizer.encode(token, add_special_tokens=False)].mean()
    sentiments.append(sentiment)

# Determine the topics using NMF
num_topics = 5
nmf = NMF(n_components=num_topics)
W = nmf.fit_transform(topic_representations)
H = nmf.components_

# Get the top words for each topic
num_words = 6
topic_words = []
for topic in H:
    topic_words.append(", ".join([tokenizer.convert_ids_to_tokens(i) for i in topic.argsort()[:-num_words-1:-1]]))

# Add the sentiment and topic to the DataFrame
df["Sentiment"] = sentiments
df["Topic"] = topic_words

# Write the results to a CSV file
df.to_csv("results.csv", index=False)
