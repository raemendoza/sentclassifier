
## Using Reddit Thread Data to create a sentence classifier

###  Introduction

Greetings! Welcome to Rae's (<https://github.com/raemendoza>) Classifier Tutorial. This tutorial is a walkthrough on how to extract online-text data to create a sentence classifier! For the sake of simplicity, I will focus on discriminating between **formal** text and **informal** text. In other words, we will train a classifier to be able to differentiate formal vs informal language. For this tutorial, we will be focusing on **two main tools**: PRAW (<https://praw.readthedocs.io/en/stable/>) and NLTK (<https://www.nltk.org>). PRAW will be used to extract the text data for us to train within NLTK and process test data. 

**For the example code and usages, please refer to [the repository for this website](https://github.com/raemendoza/raemendoza.github.io)!**

You can use the repository to refer to the code used here in full as well as the data we used to run the tests. 

**Requirements**:
Before we begin, make sure that you have the following set up and running:

1. [Python 3.9](https://www.python.org/downloads/) or higher 
2. [Pycharm](https://www.jetbrains.com/pycharm/download/#section=windows) IDE for writing and implementation
3. (*Optional*) You may choose to [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the repository through GitHub if you wish.

###  What is PRAW?

[PRAW](https://praw.readthedocs.io/en/stable/) is a python package that can make requests to the Reddit API for wrapping and data extraction. It is very versatile with the amount of data you are interested in. There are a few things you need to setup within PRAW if you wish to utilize it within your virtual environment. Once you have [configured your Pycharm's project environment](https://www.jetbrains.com/help/pycharm/setting-up-your-project.html), you need to setup the credentials needed to authenticate into Reddit's API via OAuth. Instructions can be found within PRAW's site [here](https://praw.readthedocs.io/en/stable/getting_started/authentication.html#oauth). Once you have done this, you will be provided a set of credentials. We will be implementing them into our python script for access. 

###  Authenticate to Reddit's API

Create a python script in your PyCharm project and create a new python file called `credentials.py`. Within it, insert your reddit credentials into the variable reddit **(Do not use the same credentials as the ones below, as they are sample credentials from PRAW's documentation)**.
```python
# Example code (not real credentials). You must use real credentials for this to work.
reddit = {
	"client_id": "SI8pN3DSbt0zor",
	"client_secret": "xaxkj7HNh8kwg8e5t4m6KvSrbTI", 
	"user_agent": "testscript by u/fakebot3", 
}
```

Now we need to add in all of our packages for use in these files. Find the python terminal in PyCharm and run the following command
``` python
pip install praw
pip install nltk
pip install pandas # We will use this for data-frame creation and use
```
Alternatively, you can manually install them in PyCharm's Python Packages module. See [here](https://www.jetbrains.com/help/pycharm/installing-uninstalling-and-upgrading-packages.html#install-in-tool-window) for more.

Okay, let's start extracting data with PRAW!


###  Extract Data with PRAW
First, let's extract our credentials we made earlier. Create another python file within the same directory under the name `extractor.py` where we will be writing the test of our code. We can then import the reddit credentials through the following example code:
```python
import praw  
from credentials import reddit  
  
print("Credentials imported successfully")  
  
reddit_client = praw.Reddit(**reddit)
```

Feel free to run this simple code to see if the print statement goes through successfully.
Let's think out a subreddit to use to extract our training data. **For formal text**, this should be a subreddit with the purpose of thorough discussions, whereas **informal text** should be more within casual and lighthearted topics.

Let's use [r/todayilearned](https://www.reddit.com/r/todayilearned/) for formal text and [r/funny](https://www.reddit.com/r/funny/) for informal text. 

**r/TodayILearned** is a subreddit where people write threads describing something new they discovered  briefly followed with the source of the information. People will have formal discussions about the topic in the comments.

**r/funny** speaks for itself. It's a subreddit to share humorous posts online. The comments of the discussions are a lot more brief, full of slang, and overall casual. We will use these two subreddits to train our classifier on formal and informal online text.

This example code will find the ten hottest posts at the moment from both subreddits and extract comments while expanding up to five "more" buttons in the comment threads so that it can continue to read through the comments.

```python
# Get comments from r/todayilearned  
til_comments = []  
print('Processing comments from r/todayilearned')  
til_submission = reddit_client.subreddit("todayilearned").hot(limit=10)  
for submission in til_submission:  
	submission.comments.replace_more(limit=5)  
	for comment in submission.comments.list():  
		til_comments.append(comment.body)  
	  
# Get comments from r/funny  
funny_comments = []  
print('Processing comments from r/funny')  
funny_submission = reddit_client.subreddit("funny").hot(limit=20)  
for submission in funny_submission:  
	submission.comments.replace_more(limit=5)  
	for comment in submission.comments.list():  
		funny_comments.append(comment.body)
```

This is a good start, but it would help if we could see what our comments look like so we can figure out what it looks like before pre-processing. Let's print them out.
```python
# Print the comments  
print("r/todayilearned comments:")  
for comment in til_comments:  
print(comment)  
print()  
  
print("r/funny comments:")  
for comment in funny_comments:  
print(comment)  
print()
```

Neat! This is very real-looking text data. It can be a little bit off-the-rails in language at times, but this is true data found within the realms of reddit. We must make use of it as best as we can. However, there is an issue. I notice there is a lot of noise in this data. 

 1. **URLs** will not help us with our model and will introduce noise.
 2. **Stop words** such as 'the', 'a', 'an' cloud our data with noise, and although formal speech may use them more often, the rest of the text may prove much more essential and sufficient for our classifier.
 3. **Lemmatization and stemming** will allow us to reduce words to their root form, so that we can identify similarities across words of the same lemma / stem and allow our model to contextualize these words when creating our classifier. Let's use **lemmatizer**.
 4. Some of the main comments are by **bots and moderators**, and contain sample text about the rules and regulations of the subreddit. As such, we need to remove them as well.

How will we perform all these pre-processing tasks in our extraction...? 

**NLTK TO THE RESCUE!!!**

### Pre-Processing Data

One of the best parts about NLTK is its ability to not only create classifiers, but it also has pre-processing features included. We will be importing some other packages as well to help us out, so let's **add the import statements** needed to **the top of the script** alongside our PRAW and credentials packages:
```python
#import praw 
#from credentials import reddit
#(add the code below these import statements)
import re
import string
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
```
Now let's modify our previous extraction data so it runs all the pre-processing tasks prior to creating our comment lists. We can do this in a nicely condensed pre-processor through a class of functions.
Let's add the following class code to the bottom of our `reddit_client` variable and above our reddit extractions:

```python
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
        tokenized_sentences = [word_tokenize(sentence.translate(str.maketrans('', '', string.punctuation))) for sentence in sentences]
        # Remove stop words
        tokenized_sentences = [[word for word in sentence if word.lower() not in self.stopwords] for sentence in tokenized_sentences]
        # Lemmatize words
        tokenized_sentences = [[self.lemmatizer.lemmatize(word) for word in sentence] for sentence in tokenized_sentences]
        return tokenized_sentences
``` 
Great! This class will call all our pre-processing tools, then it will pre-process the text. The *regular expression* package `re` is a syntax used to find matching patterns of text. To keep it simple, this regular expression is searching for any single string starting in `http` or `www` and replacing it with nothing. The remaining lines of code  are tokenizing, removing punctuations, stop words, and finally, lemmatizing the words as found (Keep in mind lemmatizing is not perfect, [see why here](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html).

Now, let's modify our previous code for extracting subreddit and implement the class `TextProcessor`. I will comment in what we added so you can see how our processor was implemented. We also put the moderator check in here instead since it uses PRAW to check authors.

```python
til_comments = []
print('Processing comments from r/todayilearned')
til_submission = reddit_client.subreddit("todayilearned").hot(limit=10)
for submission in til_submission:
    submission.comments.replace_more(limit=5)
    for comment in submission.comments.list():
        if comment.author != 'AutoModerator': # <- our moderator check
			til_comments.extend(TextPreprocessor().preprocess_text(comment.body)) # <- process the text

# Get comments from r/funny
funny_comments = []
print('Processing comments from r/funny')
funny_submission = reddit_client.subreddit("funny").hot(limit=20)
for submission in funny_submission:
    submission.comments.replace_more(limit=5)
    for comment in submission.comments.list():
        if comment.author != 'AutoModerator':  # <- our moderator check
			funny_comments.extend(TextPreprocessor().preprocess_text(comment.body)) # <- process the text

# Remove any token sequences that became empty as a result of processing
til_comments_tokenized = [sentence for sentence in til_comments if sentence]
funny_comments_tokenized = [sentence for sentence in funny_comments if sentence]
```

Excellent, running this with our print statements afterwards yields some nice pre-processed data!
Here's some examples of my r/TIL comments, or **formal data**.

```
['couldnt', 'find', 'copy', 'reading', 'article', 'actually', 'emailed']
['excited', 'thought', 'Id', 'writing', 'positive', 'light', 'sent', 'free', 'copy']
['Instead', 'wrote', 'framework', 'lunatic', 'conspiracy', 'theory']
```
And here is my **informal data.**
```
['cant', 'uphold']
['yes', 'banning', 'original', 'content', 'created', 'AI', 'tool', 'make', 'way', 'boomer', 'take']
['least', 'notice', 'character', 'comic', '6', 'finger']
```

To us, this may be hard to discern entirely, but this is exactly what our classifier will need to create a model! Speaking of, let's do that! To keep things simple across python scripts, we will save our token data into what we call `pickle` files. Pickle files will helps us save our data and read them on other scripts! Import it into our import statements as we did before:
```python
import pickle
# ...
# rest of code
# ...
# after creating our lists

# Save tokenized comments from r/todayilearned
with open('formal_tokens.pkl', 'wb') as f:
	pickle.dump(til_comments_tokenized, f) 
	
# Save tokenized comments from r/funny  
with open('informal_tokens.pkl', 'wb') as f:
	pickle.dump(funny_comments_tokenized, f)

```
I've changed the names of the files to something that will be easy to read in our classifier. We are going to be assuming the formality from the subreddit for the sake of demonstration. However, keep in mind, *in a realistic scenario*, we would go through the data and annotate it before processing to ensure our data measures our intended concepts. You could also have another set of annotators go through the data for a second-hand check of your judgment.

Let's go ahead and continue. After running these, you will obtain the two pickle files in the same directory. **If you wish to have the files to save to a directory of your choice, you can use the `os` module.**
 
Here's an implementation (this is optional)
 
```python
import os
import pickle

# make sure the text_data directory exists
if not os.path.exists('text_data'):
    os.makedirs('text_data')

# save til_comments_tokenized
with open('text_data/formal_tokens.pkl', 'wb') as f:
    pickle.dump(formal_tokens_tokenized, f)

# save funny_comments_tokenized
with open('text_data/informal_tokens.pkl', 'wb') as f:
    pickle.dump(funny_comments_tokenized, f)
```

All-right, let's write out classifier!

### Writing a Classifier

Let's write a new file `classifier.py` in our directory where we will import the necessary packages.
```python
import nltk  
import pickle  
import random
```

Now let's load our pickle files we created earlier and turn them into token variables.
```python
# Load formal sentences from pickle file  
with open('text_data/formal_tokens.pkl', 'rb') as f:  
	formal_tokens = pickle.load(f)  
  
# Load informal sentences from pickle file  
with open('text_data/informal_tokens.pkl', 'rb') as f:  
	informal_tokens = pickle.load(f)
```

We will need to extract features for our classifier now. A classifier is trained by providing the model with a set of features to learn and understand. Depending on the concept we are interested in, we might want to consider certain types of features to include.

What features do we extract for our tokens? Well, let's consider some popular choices:

 1. **n-grams** (token sequences of n amount): let's do unigrams and bigrams.
 2. **Part-of-speech** tags (the grammatical type of the token)
 3. The **length** of the token (apple has a length of 5)
```python
def extract_features(tokens):
    unigrams = tokens
    bigrams = list(nltk.bigrams(tokens))
    pos_tags = [tag for _, tag in nltk.pos_tag(tokens)]

    features = {}
    for word in unigrams:
        features['contains({})'.format(word)] = True
    for bigram in bigrams:
        features['bigram({},{})'.format(bigram[0], bigram[1])] = True
    for pos in pos_tags:
        features['pos({})'.format(pos)] = True
    features['length'] = len(tokens)

    return features
```

Great! But there's another thing we need to control for... Let's check the length of our data.
```python
print(len(formal_tokens))  
print(len(informal_tokens))
# formal_tokens = 9697
# informal_tokens = 5523
```

I see! There are more token sequences (sentences) in the formal  data-set! Likely because comment threads in r/todayilearned tend to be longer discussions. Remember, we did a limit for 'replacemore' back in PRAW which doesn't control for comment sizes. 
```python
submission.comments.replace_more(limit=5) # <- replaces 5 MORE buttons in A Reddit thread.
```

We could choose to filter this out during the extraction, but for now let's assume you didn't do this. We'll use random sampling to filter out the size instead. Let's reduce the size of formal tokens to be the same as informal:
```python
# Filter formal tokens by length
formal_tokens_filtered = [tokens for tokens in formal_tokens if len(tokens) >= 5]

# Get a random sample of formal tokens with the same size as informal tokens
formal_tokens_sample = random.sample(formal_tokens_filtered, len(informal_tokens))
```
Okay, now let's do a descriptive statistics test on each part. This code will import the statistics package and calculate averages for both sides.
``` python
import statistics
# rest of imports and code....

avg_formal_length = sum(len(tokens) for tokens in formal_tokens_sample) / len(formal_tokens_sample)  
print("Average length of formal token sequences:", avg_formal_length)  
  
# Calculate average length of informal token sequences  
avg_informal_length = sum(len(tokens) for tokens in informal_tokens) / len(informal_tokens)  
print("Average length of informal token sequences:", avg_informal_length)
```
And we get the following output:
```
Average length of formal token sequences: 10.294224153539743
Average length of informal token sequences: 5.852616331703784
```
As we expected, the formal token sequences tend to be longer than the informal token sequences. We could check for significance between the two sequences using scipy.stats's `ttest_ind` feature for a t-test.

```python
from scipy.stats import ttest_ind
# rest of code...

# Print results of hypothesis test  
if p_value < 0.05:  
print("The means are significantly different (p < 0.05)")  
print("p-value: {:.4f}".format(p_value))  
else:  
print("The means are not significantly different (p >= 0.05)")  
print("p-value: {:.4f}".format(p_value))
```
And we get an output of:
```
The means are significantly different (p < 0.05)
p-value: 0.0000
```

Great! So the size difference is significant! Let's continue with our feature extraction. We have to use a sample of the formal tokens, but to prevent confusion, let's rewrite the variable to be `formal_tokens` again. We will also add the feature 'formal' and 'informal' so the model know which side the features are in.

``` python
formal_tokens = formal_tokens_sample

formal_feats = [(extract_features(tokens), 'formal') for tokens in formal_tokens]
informal_feats = [(extract_features(tokens), 'informal') for tokens in informal_tokens]
```

All right, let's take a look at what these features look like briefly:
```
# Formal token example
...
'bigram(logistical,invention)': True, 
   'pos(NN)': True,
   'pos(JJ)': True,
   'pos(NNP)': True,...

# Informal token example
...
'bigram(low,effort)': True,
   'pos(VB)': True,
   'pos(NN)': True,
   'length': 9'...
```

Looks good to go! Let's run our model now.
First, we need to create a training and test set. Let's have an 80:20 ratio of train to test values.
```python
# Split the feature sets into training and testing sets (80/20)  
formal_train_size = int(len(formal_feats) * 0.8)  
formal_train_set, formal_test_set = formal_feats[:formal_train_size], formal_feats[formal_train_size:]  
informal_train_size = int(len(informal_feats) * 0.8)  
informal_train_set, informal_test_set = informal_feats[:informal_train_size], informal_feats[informal_train_size:]  
  
# Concatenate formal and informal sets  
train_set = formal_train_set + informal_train_set  
random.shuffle(train_set)
```

Now, we will get to the main part, our classifier.

### Classifier and Testing our Model

The classifier we are using comes from nltk's NaiveBayesClassifier. Here's an explanation on  [Naive Bayes](https://dataaspirant.com/naive-bayes-classifier-machine-learning/#:~:text=Naive%20Bayes%20is%20a%20kind,as%20the%20most%20likely%20class.). One thing we need to be careful about is the fact some of these features may have a conditional probability of 0. Why? Because the text data has the potential to still contain non-uniform behavior such as multiple letters for a word ('soooo annoyingg') and may be marked as a 0. This will ruin our classifier, as it will nullify any probability set by having a P = 0. Let's prevent that using something called a [Laplace smoothing](https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece). 

The tl;dr? Give a set value to all zero-probabilities, and then scale and smooth out the rest of the data to compensate. This will ensure all our features have a probability greater than zero and complete the classifier.

Let's do it! We will generate a training `classifier` and then test its accuracy with our 20% sample. As both are from different sides of the feature sets combined, they will be novel and not meshed together. 

Let's run it three times to see if random variation is too strong as well (manually doing this as I need to re-sample the data).

Here's example code for the classifier!

```python
# Split combined training set into training and validation sets
train_size = int(len(train_set) * 0.8)
train_set, val_set = train_set[:train_size], train_set[train_size:]

# Train the Naive Bayes classifier using Laplace smoothing
classifier = nltk.NaiveBayesClassifier.train(train_set, estimator=nltk.LaplaceProbDist)

# Evaluate the accuracy of the classifier on the testing set
accuracy = nltk.classify.accuracy(classifier, val_set) * 100
print("Accuracy: {:.4f}".format(accuracy) + ' %')
```

Let's view the output when I run the test three times:
```
Accuracy: 82.9186%
Accuracy: 82.0136%
Accuracy: 82.6923%
```

Decent results for our base classifier!

Let's view the features and see what's driving this model the most!

``` python
# Get the top 10 most informative features with likelihood ratios
classifier.show_most_informative_features(10)
```
And we get the following data structure!

```
Most Informative Features
       contains(English) = True           formal : inform =    124.8 : 1.0
      contains(language) = True           formal : inform =     92.7 : 1.0
        contains(French) = True           formal : inform =     51.3 : 1.0
         contains(smell) = True           inform : formal =     22.0 : 1.0
       contains(England) = True           formal : inform =     18.4 : 1.0
        contains(German) = True           formal : inform =     18.3 : 1.0
          contains(risk) = True           formal : inform =     18.3 : 1.0
        contains(cowboy) = True           inform : formal =     17.7 : 1.0
         contains(laugh) = True           inform : formal =     16.7 : 1.0
       contains(Barbara) = True           inform : formal =     16.2 : 1.0
```

Interesting! The word 'English' was the most prominent feature, as was language! Perhaps there's something within these data sets that is focalized on language within r/todayilearned. As for informal, more common words like smell and laugh (expected from r/funny appear. Maybe the 'real data' approach was giving the subreddits too much credit. It's beneficial to analyze the dataset carefully (.txt file inspection), annotate potential confounds and features that deviate from your concept of interest, and then implement them as best as you can with preprocessing code. After that, the set can be re-inspected for double-checks before being sent out to train the classifier. However, this process encompasses the trial-and-error system related to classifiers!  

**You're finished!**

Here are more resources on text classification that use different packages.

[Understanding text classification in NLP with Movie Review    Example](https://www.analyticsvidhya.com/blog/2020/12/understanding-text-classification-in-nlp-with-movie-review-example-example/)   

[Machine Learning NLP Text Classification Algorithms and    Models](https://www.projectpro.io/article/machine-learning-nlp-text-classification-algorithms-and-models/523)

[Text Classification: What and Why it Matters](https://monkeylearn.com/text-classification/)


There is no perfect package! These are all tools, after all! The key is to find what works best for you and implement it correctly and with as much attention to the data as possible!

Thanks for reading!

*Tutorial by Rae Mendoza* (<https://github.com/raemendoza>)

