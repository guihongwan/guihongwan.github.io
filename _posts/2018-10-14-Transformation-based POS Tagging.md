---  
tag: NLP 
---

# Background

Transformation-based POS Tagging or Brill's Tagging    
- It draws inspiration from the rule-based and stochastic taggers    
- It is an instance of the transformation-based learning(TBL) approach to machine learning: rules are automatically induced from the data.
- Basic idea: Do a poor job first, and then use learned rules to improve things.    

TBL
- TBL is a supervised learning technique; it assumes a pre-tagged training corpus.    
- Assumption: the word depends only on its tag.
- In this article, we focus on how TBL learns rules.


How are the rules learned?
- 1. Tag the corpus with the most likely tag for each word(unigram model)
- 2. Learn transformations: change tag a to tag b when some conditions are met
- 3. Apply the learned transformations    
- Repeat step2 and step3 until it reached some stopping criterion.

# Implementation of TBL


```python
def getDataFromFile(filename):
    ''' Read all of content from thie file as a single string'''
    file = open(filename, 'r')
    
    lines = file.readlines()
    data =""
    for line in lines:
        line = line[:-1] # remove '\n'
        data += line
        data += " "
    return data
```


```python
def MOST_LIKELY_TAGS(data):
    '''
    Using the unigram to tag each word in the corpus.
    
    data: corpus, a string
    return the dictionary with word:tag
    '''
    
    data_list = data.split()
    #print('data_list:', len(data_list))
    
    # remove the repetious word_tag to get word_tag_list
    # e.g. plant_NN, closed_VBD, in_IN
    # We will use it as a key, the value will be the number of it in data_list
    word_tag_list = list(set(data_list))
    #print('word_tag_list:',len(word_tag_list))
    
    # word_tag_count_dict, key:word_tag value:count
    wordtag_count_dict = {}
    for word_tag in word_tag_list:
        wordtag_count_dict[word_tag] = data_list.count(word_tag)
    
    # calculate the most frequent tag and the corresponding frequency for a word
    # word as key, (tag, count) as value
    best_word_tagcount = {}
    for wordtag, count in wordtag_count_dict.items():
        word, tag = wordtag.split("_")
        old_tag, old_count = best_word_tagcount.get(word, ("", 0))
        
        if count > old_count:
            best_word_tagcount[word] =(tag, count)
    
    #remove count for return
    best_word_tag = {}
    for word, tagcount in best_word_tagcount.items():
        best_word_tag[word] = tagcount[0]
        
    return best_word_tag
    
def GET_BEST_INSTANCE(data, current_tag, best_k = 1):
    '''
    Use only the previous word's tag to extract the best k transformation rules to
    i. Transform A to B
    
    return best 'best_k' transformation rules for each possible transformation
    format: (goodness, (from_tag, to_tag, pre_tag))
            means transform from_tag to to_tag when previous word's tag is pre_tag. The score is goodness.
    
    '''
    
    best_k_rules = [] # final rules

    all_tags_org = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS',
    'MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB',
    'RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN',
    'VBP','VBZ','WDT','WP','WP$','WRB',
    '$','#','\'\'','``','(',')',',','.',':','-LRB-','-RRB-'] # Penn Treebank POS tagset
    
    # get correct_word, and correct_tag from data
    # correct_word should correspond to correct_tag
    correct_tag_list = data.split() # correctly tagged words
    correct_word =[]
    correct_tag = []
    for word_tag in correct_tag_list:
        word,tag = word_tag.split("_")
        correct_word.append(word)
        correct_tag.append(tag)
    
    # remove repetious to get all_tags
    all_tags = list(set(correct_tag))
    # test
    if len(all_tags) < len(all_tags_org):
        print('all_tags_org:', len(all_tags_org))
        print('all_tags:', len(all_tags))
        notag = []
        for tag in all_tags_org:
            if tag not in all_tags:
                notag.append(tag)
        print("Those tags are not in corpus:", notag)
    # test 2
    difftag = []
    for tag in all_tags:
        if tag not in correct_tag:
            difftag.append(tag)
    for tag in correct_tag:
        if tag not in all_tags:
            difftag.append(tag)
        
    if len(difftag) > 0:
        print('Different tag:',difftag)
    else:
        print('tag test pass.')
        
    # learn the rules for 'Change from_tag to to_tag'
    for from_tag in all_tags: #e.g. NN
        for to_tag in all_tags: #e.g. NNP
            if(from_tag != to_tag): # when they are same, no need to change
                
                #1. intialize num_good_trans and num_bad_trans
                num_good_trans = {} # count the times when a change is needed
                num_bad_trans = {}  # count the times when no changes is needed.
                for i in range(len(all_tags)):
                    num_good_trans[all_tags[i]] = 0
                    num_bad_trans[all_tags[i]] = 0
            
                #2. count: most important part
                # 
                for i in range(1, len(correct_word)): # from the first word to the last one
                    # need to transform. 
                    # e.g. when current one is NN, correct one is NNP
                    # three apples
                    # correct_word[i-1]: the privous word
                    # current_tag[the privous word]: the current tag for the previous word
                    # current_tag is a dictionary, word as key
                    
                    # hence, num_good_trans recodes that:
                    # The times of changing from from_tag to to_tag, when the tag of i-1 is current_tag[the privous word]
                    # e.g. the times of changing from NN to NNP,  when the previous word of 'apples' is CD.
                    if (current_tag[correct_word[i]] == from_tag ) and correct_tag[i] == to_tag:
                        num_good_trans[current_tag[correct_word[i-1]]] += 1
                        
                    # no need to transform
                    elif (current_tag[correct_word[i]] == from_tag ) and correct_tag[i] == from_tag: # when current tag == correct tag
                        num_bad_trans[current_tag[correct_word[i-1]]] += 1
                
                #3. check out the best k transformation rules:
                trans_list = [] #(goodness, (from_tag, to_tag, pre_tag))
                for i in range(len(all_tags)):
                    pre_tag = all_tags[i]
                    good_v = num_good_trans[pre_tag]
                    bad_v =  num_bad_trans[pre_tag]
                    goodness = good_v - bad_v
                    
                    if goodness > 0:
                        trans_list.append((goodness, (from_tag, to_tag, pre_tag)))
                # order according to goodness        
                if len(trans_list)>0:
                    import heapq
                    k = best_k
                    if best_k > len(trans_list):
                        k = len(trans_list)
                        
                    best_k_rule = heapq.nlargest(k, trans_list)
                    best_k_rules.append(best_k_rule)

    return best_k_rules
```


```python
filename = "POSTaggedTrainingSet.txt" # the corpus with right tags
data = getDataFromFile(filename)
my_best_tag = MOST_LIKELY_TAGS(data)
best_rules = GET_BEST_INSTANCE(data, my_best_tag)
```

    all_tags_org: 47
    all_tags: 40
    Those tags are not in corpus: ['FW', 'LS', 'SYM', 'UH', '#', '(', ')']
    tag test pass.



```python
def printRulesForTransformation(transformation, all_rules):
    for rule in all_rules:
        from_tag =  rule[0][1][0]
        to_tag = rule[0][1][1]
        pre_tag = rule[0][1][1]
        
        if transformation[0] == from_tag and transformation[1] == to_tag:
            print(rule)
```


```python
# print rules for specific transformations
transformation1 = ['NN','VB']# from, to
printRulesForTransformation(transformation1, best_rules)
transformation2 = ['VB','NN']# from, to
printRulesForTransformation(transformation2, best_rules)
```

    [(42, ('NN', 'VB', 'MD'))]
    [(16, ('VB', 'NN', 'DT'))]

The above logs show that we need change NN to VB, when the previous word is MD, like 'can'     
and VB to NN, when the previous word is DT.


<br>

Reference：Speech and Language Processing An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition, SECOND EDITION.

<br>
For reproduction, please specify：[GHWAN's website](https://guihongwan.github.io) » [Transformation-based POS Tagging](https://guihongwan.github.io/2018/10/Transformation-based-POS-Tagging/) 
