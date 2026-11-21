import math 
from collections import Counter, defaultdict
from article import ARTICLES

def tokenize(text):
    chars = [ch if ch.isalnum() else " " for ch in text.lower()]
    return "".join(chars).split()
def build_index(articles):
    index = defaultdict(dict)
    lengths = {}
    for doc_id, (title, text) in enumerate(articles):
        words = tokenize(title + " " + text)
        lengths[doc_id] = len(words)
        for word in words:
            index[word][doc_id] = index[word].get(doc_id, 0) + 1
    return index, lengths
INDEX, LENGTHS = build_index(ARTICLES)
N = len(ARTICLES)
def score_count(query):
    scores = Counter()
    for term in tokenize(query):
        for doc_id, count in INDEX.ger(term, {}).items():
            scores[doc_id] += count
    return scores

def score_tf(query):
    scores = Counter()
    for term in tokenize(query):
        for doc_id, count in INDEX.ger(term, {}).items():
            scores[doc_id] += count / LENGTHS[doc_id]
    return scores

def idf(term):
    docs_with_term = len(INDEX.get(term,{}))
    if docs_with_term == 0:
        return 0
    return math.log(N / docs_with_term)

def score_tfidf(query):
    scores = Counter()
    for term in tokenize(query):
        for doc_id, count in INDEX.ger(term, {}).items():
                scores[doc_id] += count / LENGTHS[doc_id]* idf(term)
    return scores

RANKERS = {"COUNT": score_count, "TF": score_tf, "TF-IDF": score_tfidf}

def search(query, ranker= "TF-IDF", k=3):
    scores = RANKERS[ranker](query)
    ranked = sorted(scores.items(), key = lambda pair: pair[1], reverse = True) 
    return [(ARTICLES[doc_id][0], score) for doc_id, score in ranked[:k] if score > 0]

def snippet(title, query, bold=False):
    text = dict(ARTICLES)[title]
    terms = set(tokenize(query))
    sentences = text.replace("; ", ". ").split(". ")
    best = max(sentences, key=lambda s: sum(idf(w) for w in set(tokenize(s)) & terms))
    marked = []
    for word in best.split():
        if word.lower().split(".,;:") in terms:
            word = f"**{word}**" if bold else word.upper()
        marked.append(word)
    return " ".join(marked)

JUDGED_QUERIES = {
    ("the history of pizza", "The History of Pizza")
    ()
}