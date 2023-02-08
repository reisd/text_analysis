import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.cluster import KMeansClusterer, euclidean_distance

nltk.download("stopwords")
nltk.download("punkt")

def extract_topics(comments):
    # Tokenize comments
    words = [word_tokenize(comment.lower()) for comment in comments]

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    words = [[word for word in comment if word not in stop_words] for comment in words]

    # Get the frequency distribution of words
    fdist = FreqDist([word for comment in words for word in comment])

    # Extract the most common words
    common_words = [word[0] for word in fdist.most_common(100)]

    # Cluster words using K-means
    kmeans = KMeansClusterer(10, euclidean_distance, repeats=25)
    clusters = kmeans.cluster(common_words, assign_clusters=True)

    # Extract the top word in each cluster
    topics = [common_words[cluster.index(True)] for cluster in zip(*clusters)]

    return topics

comments = [
    "The product was not as described",
    "The service was slow and unprofessional",
    "I received the wrong product",
    "The product arrived damaged",
    "I am very satisfied with the product",
    "The customer service was excellent",
    "The product was delivered on time",
    "I will not recommend this product to others",
    "The price was too high",
    "The product was of good quality"
]

topics = extract_topics(comments)
print("Topics:", topics)
