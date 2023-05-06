import praw
from credentials import reddit
import re
import string
import pickle
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print("Credentials imported successfully")

reddit_client = praw.Reddit(**reddit)

class TextPreprocessor:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.sent_tokenizer = PunktSentenceTokenizer()

    def preprocess_text(self, text):
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\S+', '', text)
        # Tokenize into sentences
        sentences = self.sent_tokenizer.tokenize(text)
        # Tokenize each sentence into words and remove punctuation
        tokenized_sentences = [word_tokenize(sentence.translate(str.maketrans('', '', string.punctuation))) for sentence
                               in sentences]
        # Remove stop words
        tokenized_sentences = [[word for word in sentence if word.lower() not in self.stopwords] for sentence in
                               tokenized_sentences]
        # Lemmatize words
        tokenized_sentences = [[self.lemmatizer.lemmatize(word) for word in sentence] for sentence in
                               tokenized_sentences]
        return tokenized_sentences


# Get comments from r/todayilearned
til_comments = []
print('Processing comments from r/todayilearned')
til_submission = reddit_client.subreddit("todayilearned").hot(limit=10)
for submission in til_submission:
    submission.comments.replace_more(limit=5)
    for comment in submission.comments.list():
        if comment.author != 'AutoModerator':
            til_comments.extend(TextPreprocessor().preprocess_text(comment.body))

# Get comments from r/funny
funny_comments = []
print('Processing comments from r/funny')
funny_submission = reddit_client.subreddit("funny").hot(limit=20)
for submission in funny_submission:
    submission.comments.replace_more(limit=5)
    for comment in submission.comments.list():
        if comment.author != 'AutoModerator':
            funny_comments.extend(TextPreprocessor().preprocess_text(comment.body))

til_comments_tokenized = [sentence for sentence in til_comments if sentence]
funny_comments_tokenized = [sentence for sentence in funny_comments if sentence]

# Print the comments (Uncomment if you wish to preview)
# print("r/todayilearned comments:")
# for comment in til_comments:
#     print(comment)
#     print()
#
# print("r/funny comments:")
# for comment in funny_comments:
#     print(comment)
#     print()

with open('text_data/formal_tokens.pkl', 'wb') as f:
    pickle.dump(til_comments_tokenized, f)

# Save tokenized comments from r/funny
with open('text_data/informal_tokens.pkl', 'wb') as f:
    pickle.dump(funny_comments_tokenized, f)

# Save tokenized sentences as text files (uncomment if you wish to use)
with open('text_data/formal_tokenized.txt', 'w', encoding='utf-8') as file:
    file.write('\n'.join([' '.join(sent) for sent in til_comments_tokenized]))

with open('text_data/informal_tokenized.txt', 'w', encoding='utf-8') as file:
    file.write('\n'.join([' '.join(sent) for sent in funny_comments_tokenized]))

