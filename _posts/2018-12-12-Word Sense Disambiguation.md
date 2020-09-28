---  
tag: NLP 
---

# Problem
Given a sentence and a word in the sentence, we want to know what the sense of the word is.    
For example, what is the sense of $\textbf{bass}$ in 'An electric guitar and bass player stand off to one side.'?    
There are serveral senses for $\textbf{bass}$.


```python
from nltk.corpus import wordnet as wn

synsets = wn.synsets('bass')
for synset in synsets:
    print(synset, synset.definition())
```

    Synset('bass.n.01') the lowest part of the musical range
    Synset('bass.n.02') the lowest part in polyphonic music
    Synset('bass.n.03') an adult male singer with the lowest voice
    Synset('sea_bass.n.01') the lean flesh of a saltwater fish of the family Serranidae
    Synset('freshwater_bass.n.01') any of various North American freshwater fish with lean flesh (especially of the genus Micropterus)
    Synset('bass.n.06') the lowest adult male singing voice
    Synset('bass.n.07') the member with the lowest range of a family of musical instruments
    Synset('bass.n.08') nontechnical name for any of numerous edible marine and freshwater spiny-finned fishes
    Synset('bass.s.01') having or denoting a low vocal or instrumental range


# Naive Bayes Classifier
We can use supervised machine learning paradigms to solve this problem, such as Naive Bayes Classifier.    
Training data: sense taged corpus.    
One of typical features is Bag-of-words feature.    
For example, for the previous problem,     
we can have [fishing, big, player, guitar, rod] as Bag-of-words.    
Then the corresponding feature vector is [0,0,1,1,0].    

Hence, we have training data, feature vector:f, and a word.    
We want to figure out the sense of the word.

## Algorithem
$ ss = argmax_{s\in S} P(s|f)$       
$= argmax_{s\in S} {p(f|s)p(s)}$      
$= argmax_{s\in S} \prod_{i=0}^{n-1}p(f_i|s)p(s)$,We assume that $f = [f_0, f_1, ...,f_{n-1}]$ and each of $f_i$ and $f_j$ are independent.       

$p(s) = {COUNT(s,w) \over COUNT(w)}$, the number of times the sense s occurs divided by the total count of the target word.    

$p(f_i \mid s) = {COUNT(f_i,s) \over COUNT(s)}$

## Semcor
352 documents from Brown corpus manually tagged for WordNet senses.



```python
from nltk.corpus import semcor
```


```python
fileids = semcor.fileids()
fileid0 = fileids[0]
fileid0
```




    'brown1/tagfiles/br-a01.xml'




```python
sents0 = semcor.sents(fileid0)
sents0
```




    [['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', 'Friday', 'an', 'investigation', 'of', 'Atlanta', "'s", 'recent', 'primary', 'election', 'produced', '``', 'no', 'evidence', "''", 'that', 'any', 'irregularities', 'took', 'place', '.'], ['The', 'jury', 'further', 'said', 'in', 'term', 'end', 'presentments', 'that', 'the', 'City', 'Executive', 'Committee', ',', 'which', 'had', 'over-all', 'charge', 'of', 'the', 'election', ',', '``', 'deserves', 'the', 'praise', 'and', 'thanks', 'of', 'the', 'City', 'of', 'Atlanta', "''", 'for', 'the', 'manner', 'in', 'which', 'the', 'election', 'was', 'conducted', '.'], ...]




```python
sem0 = semcor.tagged_sents(fileid0, tag='sem')
sem0[0] # first sentence
```




    [['The'],
     Tree(Lemma('group.n.01.group'), [Tree('NE', ['Fulton', 'County', 'Grand', 'Jury'])]),
     Tree(Lemma('state.v.01.say'), ['said']),
     Tree(Lemma('friday.n.01.Friday'), ['Friday']),
     ['an'],
     Tree(Lemma('probe.n.01.investigation'), ['investigation']),
     ['of'],
     Tree(Lemma('atlanta.n.01.Atlanta'), ['Atlanta']),
     ["'s"],
     Tree(Lemma('late.s.03.recent'), ['recent']),
     Tree(Lemma('primary.n.01.primary_election'), ['primary', 'election']),
     Tree(Lemma('produce.v.04.produce'), ['produced']),
     ['``'],
     ['no'],
     Tree(Lemma('evidence.n.01.evidence'), ['evidence']),
     ["''"],
     ['that'],
     ['any'],
     Tree(Lemma('abnormality.n.04.irregularity'), ['irregularities']),
     Tree(Lemma('happen.v.01.take_place'), ['took', 'place']),
     ['.']]




```python
tree = sem0[0][2]
```


```python
tree.label()
```




    Lemma('state.v.01.say')




```python
tree.leaves()
```




    ['said']



## Implementation


```python
def count_for_nbc(word, sense, f):
    '''
    Given a word, find out the taged-sents with word
    '''
    import numpy as np
    length = len(f)
    wf = list(np.zeros(length))
    COUNTS = list(np.zeros(4)) #(w), (s), (w,s), (w,fi)
    COUNTS[3] = wf

    from nltk.corpus import semcor
    import nltk
    fileids = semcor.fileids()
    for fd in fileids:
        sents = semcor.tagged_sents(fd, tag='sem')
        for sent in sents:
            for word in sent:
                if isinstance(word, nltk.tree.Tree):
                    lemma = word.label()
                   
                    W = False
                    S = False
                    # meet the word
                    if isinstance(lemma, nltk.corpus.reader.wordnet.Lemma) and (word == lemma.name()):
                        COUNTS[0] += 1
                        W = True
                        # in the sentence, if there is a k
                        i = 0
                        for k,v in f.items():
                            flag = isWordInSentence(k, sent)
                            if flag == v:
                                COUNTS[3][i] += 1
                            i += 1
                    # meet the sense
                    if isinstance(lemma, nltk.corpus.reader.wordnet.Lemma) and (lemma.synset() == sense):
                        COUNTS[1] += 1
                        S = True
                    if W and S:
                        COUNTS[2] += 1
                    
    return COUNTS

def isWordInSentence(word, tagged_sent):
    '''
    if word is in the sentence, return 1, otherwise 0.
    '''
    for word in tagged_sent:
        if isinstance(word, nltk.tree.Tree):
            lemma = word.label()
            if isinstance(lemma, nltk.corpus.reader.wordnet.Lemma) and (word == lemma.name()):
                return 1
        else:
            print(word)
    return 0
```


```python
import nltk
from nltk.corpus import wordnet as wn

#all of keys are lemmanized
f = {}
f['fishing'] = 0
f['big'] = 0
f['player'] = 1
f['guitar'] = 1
f['rod'] = 0
word = 'bass' # here, word is lemmanized

synsets = wn.synsets(word)
counts = {}
for synset in synsets:
    count = count_for_nbc(word, synset, f)
    counts[synset] = count
```


```python
#(w), (s), (w,s), (w,fi)
import math
import numpy as np
for s, count in counts.items():
    ret = {}
    if(count[0] != 0):
        p_s = count[2]/count[0]
        fs = count[3]
        p = 1
        for c in fs:
            p_fs = (c/count[1])*p_s
            p *= p_fs
        ret[s] = p

```

## Problem
The algorithm fails for words not in the training data.

# Lesk Algorithm
- Use dictionary or thesaurus as indirect kind of supervision.     
- Choose the sense whose gloss shares the most words with target word neighborhood.

## Implementation


```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

def simplified_lesk(target, sentence):
    '''
    Get the word(target) meaning in sentence

    :param target:
    :param sentence:
    :return:
    '''
    
    stop_words = stopwords.words('english')
    s_list = [token.lower() for token in word_tokenize(sentence)]

    sets = wn.synsets(target) # all the meanings of target
    
    maxsense = ""
    maximum = 0
    max_n = 0
    for i in range(len(sets)):
        overlap = []

        # print('1. Obtain gloss and examples')
        bank = sets[i]
        # gloss
        gloss = word_tokenize(bank.definition())
        # examples
        examples = []
        es = [word_tokenize(example) for example in bank.examples()]
        for word_list in es:
            for word in word_list:
                examples.append(word.lower())

        # print('2. Overlap')
        for token in s_list:
            if token in examples:
                overlap.append(token)
            if token in gloss:
                overlap.append(token)

        # print('3. check stopwords and tareget in Overlap')
        for word in overlap:
            if word in stop_words:
                while word in overlap:
                    overlap.remove(word)
        if target in overlap:
            while target in overlap:
                overlap.remove(target)
                
        # print('4. maximum')
        if len(overlap) > maximum:
            max_n = i
            maximum = len(overlap)
            maxsense = gloss
        
        print('Overlap for sense', i, 'is', overlap)
    chosen_sense = " ".join(maxsense)
    print()
    # print('The final chosen sense:', max_n, '\n', chosen_sense)
    return sets[max_n]
```


```python
sentence = ('The bank can guarantee deposits will eventually cover future tuition costs because it invests in adjustable-rate mortgage securities.')	
sense = simplified_lesk('bank', sentence)
print(sense, sense.definition())
```

    Overlap for sense 0 is []
    Overlap for sense 1 is ['deposits', 'mortgage']
    Overlap for sense 2 is []
    Overlap for sense 3 is []
    Overlap for sense 4 is ['future']
    Overlap for sense 5 is []
    Overlap for sense 6 is ['in']
    Overlap for sense 7 is []
    Overlap for sense 8 is []
    Overlap for sense 9 is []
    Overlap for sense 10 is []
    Overlap for sense 11 is []
    Overlap for sense 12 is []
    Overlap for sense 13 is ['in']
    Overlap for sense 14 is ['in']
    Overlap for sense 15 is ['deposits']
    Overlap for sense 16 is ['cover']
    Overlap for sense 17 is []
    
    Synset('depository_financial_institution.n.01') a financial institution that accepts deposits and channels the money into lending activities


## Problem and Improvement

- Main problem with Lesk algorithm is the small number of words in gloss definitions
- Possible improvements
  - Include related words, i.e. hyponyms    
  - Appy a weight to each overlapping word    
  $weight_i = log({Ndoc \over Ndoc(word_i)})$    
  Ndoc: number of documents in corpus.    
  Ndoc(word_i): number of documents where word_i is included.

<br>

Reference：Speech and Language Processing An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition, SECOND EDITION.

<br>
For reproduction, please specify：[GHWAN's website](https://guihongwan.github.io) » [Word Sense Disambiguation](https://guihongwan.github.io/2018/12/Word-Sense-Disambiguation/)
