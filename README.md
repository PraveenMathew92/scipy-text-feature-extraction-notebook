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

    Requirement already satisfied: nbconvert in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (6.0.7)
    Requirement already satisfied: entrypoints>=0.2.2 in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from nbconvert) (0.3)
    Requirement already satisfied: pandocfilters>=1.4.1 in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from nbconvert) (1.4.3)
    Requirement already satisfied: testpath in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from nbconvert) (0.4.4)
    Requirement already satisfied: defusedxml in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from nbconvert) (0.6.0)
    Requirement already satisfied: jupyter-core in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from nbconvert) (4.6.3)
    Requirement already satisfied: mistune<2,>=0.8.1 in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from nbconvert) (0.8.4)
    Requirement already satisfied: nbclient<0.6.0,>=0.5.0 in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from nbconvert) (0.5.1)
    Requirement already satisfied: bleach in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from nbconvert) (3.2.1)
    Requirement already satisfied: jupyterlab-pygments in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from nbconvert) (0.1.2)
    Requirement already satisfied: nbformat>=4.4 in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from nbconvert) (5.0.8)
    Requirement already satisfied: pygments>=2.4.1 in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from nbconvert) (2.7.2)
    Requirement already satisfied: traitlets>=4.2 in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from nbconvert) (5.0.5)
    Requirement already satisfied: jinja2>=2.4 in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from nbconvert) (2.11.2)
    Requirement already satisfied: nest-asyncio in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from nbclient<0.6.0,>=0.5.0->nbconvert) (1.4.2)
    Requirement already satisfied: async-generator in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from nbclient<0.6.0,>=0.5.0->nbconvert) (1.10)
    Requirement already satisfied: jupyter-client>=6.1.5 in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from nbclient<0.6.0,>=0.5.0->nbconvert) (6.1.7)
    Requirement already satisfied: six>=1.9.0 in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from bleach->nbconvert) (1.15.0)
    Requirement already satisfied: packaging in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from bleach->nbconvert) (20.4)
    Requirement already satisfied: webencodings in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from bleach->nbconvert) (0.5.1)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from nbformat>=4.4->nbconvert) (3.2.0)
    Requirement already satisfied: ipython-genutils in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from nbformat>=4.4->nbconvert) (0.2.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from jinja2>=2.4->nbconvert) (1.1.1)
    Requirement already satisfied: python-dateutil>=2.1 in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from jupyter-client>=6.1.5->nbclient<0.6.0,>=0.5.0->nbconvert) (2.8.1)
    Requirement already satisfied: pyzmq>=13 in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from jupyter-client>=6.1.5->nbclient<0.6.0,>=0.5.0->nbconvert) (19.0.2)
    Requirement already satisfied: tornado>=4.1 in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from jupyter-client>=6.1.5->nbclient<0.6.0,>=0.5.0->nbconvert) (6.0.4)
    Requirement already satisfied: pyparsing>=2.0.2 in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from packaging->bleach->nbconvert) (2.4.7)
    Requirement already satisfied: attrs>=17.4.0 in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.4->nbconvert) (20.3.0)
    Requirement already satisfied: setuptools in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.4->nbconvert) (50.3.1.post20201107)
    Requirement already satisfied: pyrsistent>=0.14.0 in /Users/mpraveen/anaconda3/lib/python3.8/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.4->nbconvert) (0.17.3)



```python
! jupyter nbconvert --help
```

    This application is used to convert notebook files (*.ipynb) to various other
    formats.
    
    WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only 
        relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place, 
        overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document. 
        This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
        ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf',
        'python', 'rst', 'script', 'slides', 'webpdf'] or a dotted object name that
        represents the import path for an `Exporter` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --writer=<DottedObjectName>
        Writer class used to write the  results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        overwrite base name use for output files. can only be used when converting
        one notebook at a time.
        Default: ''
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults to output to the directory of each
        notebook. To recover previous default behaviour (outputting to the current
        working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x). This defaults to the reveal CDN,
        but can be any url pointing to a copy  of reveal.js.
        For speaker notes to work, this must be a relative path to a local  copy of
        reveal.js: e.g., "reveal.js".
        If a relative path is given, it must be a subdirectory of the current
        directory (from which the server is run).
        See the usage documentation
        (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-
        slideshow) for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write. Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
        
        > jupyter nbconvert mynotebook.ipynb --to html
        
        Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', 'slides', 'webpdf'].
        
        > jupyter nbconvert --to latex mynotebook.ipynb
        
        Both HTML and LaTeX support multiple output templates. LaTeX includes
        'base', 'article' and 'report'.  HTML includes 'basic' and 'full'. You
        can specify the flavor of the format used.
        
        > jupyter nbconvert --to html --template lab mynotebook.ipynb
        
        You can also pipe the output to stdout, rather than a file
        
        > jupyter nbconvert mynotebook.ipynb --stdout
        
        PDF is generated via latex
        
        > jupyter nbconvert mynotebook.ipynb --to pdf
        
        You can get (and serve) a Reveal.js-powered slideshow
        
        > jupyter nbconvert myslides.ipynb --to slides --post serve
        
        Multiple notebooks can be given at the command line in a couple of 
        different ways:
        
        > jupyter nbconvert notebook*.ipynb
        > jupyter nbconvert notebook1.ipynb notebook2.ipynb
        
        or you can specify the notebooks list in a config file, containing::
        
            c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
        
        > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    



```python
! jupyter nbconvert --to markdown TextFeatureExtraction.ipynb README.md
```

    [NbConvertApp] WARNING | pattern 'README.md' matched no files
    [NbConvertApp] Converting notebook TextFeatureExtraction.ipynb to markdown
    [NbConvertApp] Writing 14388 bytes to TextFeatureExtraction.md



```python

```
