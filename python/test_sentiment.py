import re
from collections import Counter
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
data = ["I love you", "I hate you"]
print(sentiment_pipeline(data))
exit(0)

def tokenize(text):
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)

import liwc
parse, category_names = liwc.load_token_parser('LIWC2007_English100131.dic')

gettysburg = '''Four score and seven years ago our fathers brought forth on
  this continent a new nation, conceived in liberty, and dedicated to the
  proposition that all men are created equal. Now we are engaged in a great
  civil war, testing whether that nation, or any nation so conceived and so
  dedicated, can long endure. We are met on a great battlefield of that war.
  We have come to dedicate a portion of that field, as a final resting place
  for those who here gave their lives that that nation might live. It is
  altogether fitting and proper that we should do this.'''
gettysburg_tokens = tokenize(gettysburg)
# now flatmap over all the categories in all of the tokens using a generator:
gettysburg_counts = Counter(category for token in gettysburg_tokens for category in parse(token))
# and print the results:
print(gettysburg_counts)