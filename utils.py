import numpy as np
import random
import string
import matplotlib.pyplot as plt
from time import perf_counter
from datetime import timedelta
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_rcv1
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics

stopwords_long = ["a", "able", "about", "above", "abst", "accordance", "according", "accordingly", "across", "act", "actually", "added", "adj", "affected", "affecting", "affects", "after", "afterwards", "again", "against", "ah", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "apparently", "approximately", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "auth", "available", "away", "awfully", "b", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "between", "beyond", "biol", "both", "brief", "briefly", "but", "by", "c", "ca", "came", "can", "cannot", "can't", "cause", "causes", "certain", "certainly", "co", "com", "come", "comes", "contain", "containing", "contains", "could", "couldnt", "d", "date", "did", "didn't", "different", "do", "does", "doesn't", "doing", "done", "don't", "down", "downwards", "due", "during", "e", "each", "ed", "edu", "effect", "eg", "eight", "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et", "et-al", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "except", "f", "far", "few", "ff", "fifth", "first", "five", "fix", "followed", "following", "follows", "for", "former", "formerly", "forth", "found", "four", "from", "further", "furthermore", "g", "gave", "get", "gets", "getting", "give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten", "h", "had", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he", "hed", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hers", "herself", "hes", "hi", "hid", "him", "himself", "his", "hither", "home", "how", "howbeit", "however", "hundred", "i", "id", "ie", "if", "i'll", "im", "immediate", "immediately", "importance", "important", "in", "inc", "indeed", "index", "information", "instead", "into", "invention", "inward", "is", "isn't", "it", "itd", "it'll", "its", "itself", "i've", "j", "just", "k", "keep	keeps", "kept", "kg", "km", "know", "known", "knows", "l", "largely", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", "line", "little", "'ll", "look", "looking", "looks", "ltd", "m", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "more", "moreover", "most", "mostly", "mr", "mrs", "much", "mug", "must", "my", "myself", "n", "na", "name", "namely", "nay", "nd", "near", "nearly", "necessarily", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "ninety", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "now", "nowhere", "o", "obtain", "obtained", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "omitted", "on", "once", "one", "ones", "only", "onto", "or", "ord", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "owing", "own", "p", "page", "pages", "part", "particular", "particularly", "past", "per", "perhaps", "placed", "please", "plus", "poorly", "possible", "possibly", "potentially", "pp", "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud", "provides", "put", "q", "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", "re", "readily", "really", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "respectively", "resulted", "resulting", "results", "right", "run", "s", "said", "same", "saw", "say", "saying", "says", "sec", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several", "shall", "she", "shed", "she'll", "shes", "should", "shouldn't", "show", "showed", "shown", "showns", "shows", "significant", "significantly", "similar", "similarly", "since", "six", "slightly", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specifically", "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure	t", "take", "taken", "taking", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'll", "theyre", "they've", "think", "this", "those", "thou", "though", "thoughh", "thousand", "throug", "through", "throughout", "thru", "thus", "til", "tip", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "up", "upon", "ups", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "v", "value", "various", "'ve", "very", "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "was", "wasnt", "way", "we", "wed", "welcome", "we'll", "went", "were", "werent", "we've", "what", "whatever", "what'll", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "whose", "why", "widely", "willing", "wish", "with", "within", "without", "wont", "words", "world", "would", "wouldnt", "www", "x", "y", "yes", "yet", "you", "youd", "you'll", "your", "youre", "yours", "yourself", "yourselves", "you've", "z", "zero"]

stem = PorterStemmer()
wnl = WordNetLemmatizer()
def preprocessing_(doc):
    doc = doc.lower()
#     doc = remove_metadata(doc)
    
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    
    doc = word_tokenize(doc)
    
    doc = filter(lambda x:x not in string.punctuation, doc)

    doc = filter(lambda x:x not in stopwords_long, doc)
    
    doc = filter(lambda x:not x.isdigit(), doc)
    doc = [wnl.lemmatize(w.lower()) for w in doc]
    doc = [stem.stem(w) for w in doc]
    doc = ' '.join(e for e in doc)
#     print(doc)
    return doc

def preprocessing_nostem(doc):
    doc = doc.lower()
#     doc = remove_metadata(doc)
    
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    
    doc = word_tokenize(doc)
    
    doc = filter(lambda x:x not in string.punctuation, doc)

    doc = filter(lambda x:x not in stopwords_long, doc)
    
    doc = filter(lambda x:not x.isdigit(), doc)
    doc = [wnl.lemmatize(w.lower()) for w in doc]
    #doc = [stem.stem(w) for w in doc]
    doc = ' '.join(e for e in doc)
#     print(doc)
    return doc


def read_data(group, total=200):
    
    if group == 'NG3':
        NG = ['comp.graphics','rec.sport.baseball','talk.politics.guns']
    elif group == 'NG6':
        NG = ['alt.atheism','comp.sys.mac.hardware', 'rec.motorcycles', 'rec.sport.hockey','soc.religion.christian',
             'talk.religion.misc']
    else:
        NG = ['talk.politics.mideast', 'talk.politics.misc', 'comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware',
              'sci.electronics', 'sci.crypt', 'sci.med', 'sci.space', 'misc.forsale']
        
    text_corpus = []
    file_names = []
    target = np.arange(0,len(NG)).tolist()*total
    target.sort()
    for i,category in enumerate(NG):
        np.random.seed(i+42)
        news = fetch_20newsgroups(subset='train',categories=[category])
        permutation = np.arange(len(news.data)).tolist()
        np.random.shuffle(permutation)
        permutation = random.sample(permutation,total)
        randomtext_200 = np.asarray(news.data)[permutation]
        files_200 = news.filenames[permutation]
        text_corpus = text_corpus + randomtext_200.tolist()
        file_names = file_names + files_200.tolist()
        
    return text_corpus, file_names, target

    
def get_cosine_sim_matrix(text_corpus, mindf=1):
    
    vectorizer = TfidfVectorizer(min_df=mindf)
    vectors = vectorizer.fit_transform(text_corpus)
    cosine_sim_matrix = cosine_similarity(vectors)
    
    return cosine_sim_matrix

def compute_metrics(embeddings, target):
    
    clf = MultinomialNB(alpha=0.1)
    clf.fit(embeddings, target)
    preds = clf.predict(embeddings)
    f1 = metrics.f1_score(target, preds, average='macro')
    
    
    print("\nEvaluated embeddings using Multinomial Naive Bayes")
    print("F1 - score(Macro) : ",f1)
    
    
    return f1

def scale_sim_matrix(mat):
    #Row-wise sacling of matrix
    mat = mat - np.diag(np.diag(mat)) #Make diag elements zero
    D_inv = np.diag(np.reciprocal(np.sum(mat,axis=0)))
    mat = np.dot(D_inv, mat)
    
    return mat