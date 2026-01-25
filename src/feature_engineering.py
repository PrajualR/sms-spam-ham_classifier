# from nltk.tokenize import word_tokenize
# from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# def word2vec_model(df, vector_size=100, window =5, min_count=1):
#     """Train a Word2Vec model on SMS dataset"""
#
#     tokenized_messages = [word_tokenize(message) for message in df['clean_message']]
#
#     w2v_model = Word2Vec(sentences=tokenized_messages, vector_size=vector_size, window=window, min_count=min_count, workers=4)
#
#     return w2v_model
#
# def get_avg_word2vec(tokens, model, vector_size):
#     """Convert a list of tokens to an averaged word vector"""
#     valid_tokens = [token for token in tokens if token in model.wv.key_to_index]
#
#     if len(valid_tokens) == 0:
#         # Return zero vector if no tokens are in vocabulary
#         return np.zeros(vector_size)
#
#     # Sum up embeddings for all tokens and take average
#     word_vectors = np.array([model.wv[token] for token in valid_tokens])
#     return np.mean(word_vectors, axis=0)
#
# def create_avgword2vec(df,model, vector_size=100):
#     """Create AvgWord2Vec features for each message"""
#     features=[]
#     for message in df['clean_message']:
#         tokens = word_tokenize(message)
#         avg_vector = get_avg_word2vec(tokens, model, vector_size)
#         features.append(avg_vector)
#
#     return np.array(features)

def create_tfidf_features(df, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(3, 5), min_df=2, analyzer="char")
    X_tfidf = vectorizer.fit_transform(df['clean_message'])
    return X_tfidf, vectorizer

def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)