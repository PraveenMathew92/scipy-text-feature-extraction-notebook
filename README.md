### Following the scikit learn blog on Text Feature Extraction
https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

## Count Vectoroizer


```python
from sklearn.feature_extraction.text import CountVectorizer
```

Create corpus(a list of text documents)


```python
corpus = [
    'Thrice to thine',
    'and Thrice to mine',
    'and thrice again to make it nine'
]
```


```python
vectorizer = CountVectorizer()
vectorizer.fit(corpus) # learn the vocabulary in the corpus

print(list(enumerate(vectorizer.get_feature_names()))) # get the vocabulary
print(vectorizer.transform(corpus).toarray()) # DTM: Document Term Matrix
```

    [(0, 'again'), (1, 'and'), (2, 'it'), (3, 'make'), (4, 'mine'), (5, 'nine'), (6, 'thine'), (7, 'thrice'), (8, 'to')]
    [[0 0 0 0 0 0 1 1 1]
     [0 1 0 0 1 0 0 1 1]
     [1 1 1 1 0 1 0 1 1]]



```python
bi_gram_Vector = CountVectorizer(ngram_range=(2, 2))
bi_gram_document_term_matrix = bi_gram_Vector.fit_transform(corpus)

analyser = bi_gram_Vector.build_analyzer()
preprocessor = bi_gram_Vector.build_preprocessor()
tokenizer = bi_gram_Vector.build_tokenizer()

text = "What goes around comes around"

print('tokenizer : ', tokenizer(text))
print('preprocessor : ', preprocessor(text))
print('analyser : ', analyser(text))
```

    tokenizer :  ['What', 'goes', 'around', 'comes', 'around']
    preprocessor :  what goes around comes around
    analyser :  ['what goes', 'goes around', 'around comes', 'comes around']



```python
print(list(enumerate(bi_gram_Vector.get_feature_names()))) # get the vocabulary
print(bi_gram_Vector.transform(corpus).toarray()) # DTM: Document Term Matrix
```

    [(0, 'again to'), (1, 'and thrice'), (2, 'it nine'), (3, 'make it'), (4, 'thrice again'), (5, 'thrice to'), (6, 'to make'), (7, 'to mine'), (8, 'to thine')]
    [[0 0 0 0 0 1 0 0 1]
     [0 1 0 0 0 1 0 1 0]
     [1 1 1 1 1 0 1 0 0]]


## TF-IDF Transformer


```python
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
```


```python
pipeline = make_pipeline(CountVectorizer(), TfidfTransformer())
tf_idf = pipeline.fit_transform(corpus)
tf_idf.toarray()
```




    array([[0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.76749457, 0.45329466, 0.45329466],
           [0.        , 0.50410689, 0.        , 0.        , 0.66283998,
            0.        , 0.        , 0.39148397, 0.39148397],
           [0.43535684, 0.3311001 , 0.43535684, 0.43535684, 0.        ,
            0.43535684, 0.        , 0.25712876, 0.25712876]])




```python
features = pipeline.named_steps.countvectorizer.get_feature_names()
idf = pipeline.named_steps.tfidftransformer.idf_
set(zip(features, idf)) # idf = ln((n+1)/(df(t)+1)) + 1
```




    {('again', 1.6931471805599454),
     ('and', 1.2876820724517808),
     ('it', 1.6931471805599454),
     ('make', 1.6931471805599454),
     ('mine', 1.6931471805599454),
     ('nine', 1.6931471805599454),
     ('thine', 1.6931471805599454),
     ('thrice', 1.0),
     ('to', 1.0)}



### IDF if smoothen was false


```python
pipeline.named_steps.tfidftransformer.set_params(smooth_idf = False)
tf_idf = pipeline.fit_transform(corpus)

features = pipeline.named_steps.countvectorizer.get_feature_names()
idf = pipeline.named_steps.tfidftransformer.idf_
set(zip(features, idf)) # idf = ln(n/df(t)) + 1
```




    {('again', 2.09861228866811),
     ('and', 1.4054651081081644),
     ('it', 2.09861228866811),
     ('make', 2.09861228866811),
     ('mine', 2.09861228866811),
     ('nine', 2.09861228866811),
     ('thine', 2.09861228866811),
     ('thrice', 1.0),
     ('to', 1.0)}




```python
tf_idf.toarray() 
```




    array([[0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.829279  , 0.39515588, 0.39515588],
           [0.        , 0.48552418, 0.        , 0.        , 0.72497497,
            0.        , 0.        , 0.34545446, 0.34545446],
           [0.45163284, 0.30246377, 0.45163284, 0.45163284, 0.        ,
            0.45163284, 0.        , 0.21520547, 0.21520547]])



### TF-IDF without Normalizer


```python
pipeline.named_steps.tfidftransformer.set_params(norm='')
tf_idf = pipeline.fit_transform(corpus)
tf_idf.toarray() 
```




    array([[0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 2.09861229, 1.        , 1.        ],
           [0.        , 1.40546511, 0.        , 0.        , 2.09861229,
            0.        , 0.        , 1.        , 1.        ],
           [2.09861229, 1.40546511, 2.09861229, 2.09861229, 0.        ,
            2.09861229, 0.        , 1.        , 1.        ]])



### TF-IDF with L1 Normalizer


```python
pipeline.named_steps.tfidftransformer.set_params(norm='l1')
tf_idf = pipeline.fit_transform(corpus)
tf_idf.toarray() 
```




    array([[0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.51202996, 0.24398502, 0.24398502],
           [0.        , 0.25534981, 0.        , 0.        , 0.38128321,
            0.        , 0.        , 0.18168349, 0.18168349],
           [0.17784979, 0.11910808, 0.17784979, 0.17784979, 0.        ,
            0.17784979, 0.        , 0.08474638, 0.08474638]])



## Normalizers in TD-IDF
### L2:  **(default)** 
*Formula* = $ \frac{v_i}{\sqrt\Sigma(v_i^2)}$

### L1:
*Formula* = $ \frac{v_i}{\Sigma |v_i|}$

------

## TF IDF Vectorizer

It is a combination of the Count Vectorizer and the TF-IDF Transformer


```python
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_vectoriser = TfidfVectorizer()
document_term_matrix = tf_idf_vectoriser.fit_transform(corpus)
features = tf_idf_vectoriser.get_feature_names()

for document in list(enumerate(document_term_matrix.toarray())):
    print("DOCUMENT NUMBER ", document[0])
    for token, tf_idf in zip(features, document[1]):
        if(tf_idf > 0):
            print(f'\t {token:10} : {tf_idf}')
```

    DOCUMENT NUMBER  0
    	 thine      : 0.7674945674619879
    	 thrice     : 0.4532946552278861
    	 to         : 0.4532946552278861
    DOCUMENT NUMBER  1
    	 and        : 0.5041068915759233
    	 mine       : 0.6628399823470976
    	 thrice     : 0.39148397136265967
    	 to         : 0.39148397136265967
    DOCUMENT NUMBER  2
    	 again      : 0.43535684236960664
    	 and        : 0.33110010014200913
    	 it         : 0.43535684236960664
    	 make       : 0.43535684236960664
    	 nine       : 0.43535684236960664
    	 thrice     : 0.25712876433201076
    	 to         : 0.25712876433201076


Similar count may be obtained for 2-5 gram words with no normalization


```python
tf_idf_vectoriser = TfidfVectorizer(ngram_range=(2, 5), norm='')
document_term_matrix = tf_idf_vectoriser.fit_transform(corpus)
features = tf_idf_vectoriser.get_feature_names()

for document in list(enumerate(document_term_matrix.toarray())):
    print("DOCUMENT NUMBER ", document[0])
    for token, tf_idf in zip(features, document[1]):
        if(tf_idf > 0):
            print(f'\t {token:10} : {tf_idf}')
```

    DOCUMENT NUMBER  0
    	 thrice to  : 1.2876820724517808
    	 thrice to thine : 1.6931471805599454
    	 to thine   : 1.6931471805599454
    DOCUMENT NUMBER  1
    	 and thrice : 1.2876820724517808
    	 and thrice to : 1.6931471805599454
    	 and thrice to mine : 1.6931471805599454
    	 thrice to  : 1.2876820724517808
    	 thrice to mine : 1.6931471805599454
    	 to mine    : 1.6931471805599454
    DOCUMENT NUMBER  2
    	 again to   : 1.6931471805599454
    	 again to make : 1.6931471805599454
    	 again to make it : 1.6931471805599454
    	 again to make it nine : 1.6931471805599454
    	 and thrice : 1.2876820724517808
    	 and thrice again : 1.6931471805599454
    	 and thrice again to : 1.6931471805599454
    	 and thrice again to make : 1.6931471805599454
    	 it nine    : 1.6931471805599454
    	 make it    : 1.6931471805599454
    	 make it nine : 1.6931471805599454
    	 thrice again : 1.6931471805599454
    	 thrice again to : 1.6931471805599454
    	 thrice again to make : 1.6931471805599454
    	 thrice again to make it : 1.6931471805599454
    	 to make    : 1.6931471805599454
    	 to make it : 1.6931471805599454
    	 to make it nine : 1.6931471805599454


---

---
# Text Processing Notes


https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- Algorithms require numerical feature vector
- Convert raw string to numerical feature vector : vectorization
    - Bag of n grams:
        - Tokenization: convert string to tokens
        - Counting: count the number of tokens in each vector
        - Normalization: give weight to the tokens present in the document
- Steps in vectorizer class:
    - Preprocessor: clean the string like remove html tags and convert all to lowercase
    - Tokenizer: preprocessor outputs => tokens.
    - Analyzer: by default calls the preprocessor, tokenizer and does n-gram extraction and stop word filtering.


https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
Count Vectorizer:
- Text document to matrix of token count


https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
TFIDF Transformer
- Count matrix => TF-IDF representation


https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
https://scikit-learn.org/stable/modules/compose.html#pipeline
Pipelines in skit learn
- Pipeline has steps
- Each step has a name and an estimator object.
- Pipeline is a list of tuple of name string and estimator object
- Calling fit on the pipeline is same as calling fit and transform on each and every estimator in the pipeline


```python
! pip install nbconvert
```


```python
! jupyter nbconvert --to markdown TextFeatureExtraction.ipynb
! mv TextFeatureExtraction.md README.md
```

    [NbConvertApp] Converting notebook TextFeatureExtraction.ipynb to markdown
    [NbConvertApp] Writing 21653 bytes to TextFeatureExtraction.md



```python

```
