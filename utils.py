import time
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
start_time = time.time()
stop_words = set(stopwords.words('english'))

# Our own list of some block words to be avoided
block_words = ['newsgroups', 'xref', 'path', 'from', 'subject', 'sender', 'organisation', 'apr', 'gmt', 'last',
               'better', 'never', 'every', 'even', 'two', 'good', 'used', 'first', 'need', 'going', 'must',
               'really', 'might', 'well', 'without', 'made', 'give', 'look', 'try', 'far', 'less', 'seem', 'new', 'make',
               'many', 'way', 'since', 'using', 'take', 'help', 'thanks', 'send', 'free', 'may', 'see', 'much', 'want',
               'find', 'would', 'one', 'like', 'get', 'use', 'also', 'could', 'say', 'us', 'go', 'please', 'said', 'set',
               'got', 'sure', 'come', 'lot', 'seems', 'able', 'anything', 'put', '--', '|>', '>>', '93', 'xref',
               'cantaloupe.srv.cs.cmu.edu', '20', '16', "max>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'",
               '21', '19', '10', '17', '24', 'reply-to:', 'thu', 'nntp-posting-host:', 're:',
               '25''18'"i'd"'>i''22''fri,''23''>the', 'references:', 'xref:', 'sender:', 'writes:', '1993',
               'organization:']
