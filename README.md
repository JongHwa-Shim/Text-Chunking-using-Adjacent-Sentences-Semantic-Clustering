# Text Chunking using Adjacent Sentences Semantic Clustering (인접 문장 의미론적 군집화를 사용한 텍스트 청킹)
- Address text chunks(adjacent sentences) as semantic clusters.
- Clustering texts based on the similarity between adjacent sentences → each cluster becomes a text chunk.
- Using the Sentence Transformer as a vectorization model for similarity calculation, **implementation possible even in closed networks**
- Adjusting the similarity threshold or minimum and maximum chunk sizes allows arbitrary **chunking environment control**
- **Pros:** Content consistency of each chunk, flexible chunk size, adjustable parameters such as threshold, order preservation
- **Cons:** The method is somewhat complex, increases processing time, and is not effective if chunk size uniformity is important. If the text content is very complex (mixed with sentences, word fragments, and formatting symbols), the vectorization performance of the sentence transformer may degrade.

## Chunking Comparison
- This program address text chunks as semantic clusters and can devide text more semantically appropriately.
### 1. Sample chunks from 'Langchain Splitter with Custom Parameters':

```
Chunk 1: 
Brazil is the world's fifth-largest country by area and the seventh most popul ous. Its capital is Brasília, and its most popul ous city is São Paulo. The federation is composed of the union of the 26

Chunk 2: 
states and the Federal District. It is the only country in the Americas to have Portugue se as an official language.[11][12] It is one of the most multicultural and ethnically diverse nations, due to over a century of
```

### 2. (Ours) Sample chunks from 'Adjacent Sentences Semantic Clustering':

```
Chunk 1:
Brazil is the world's fifth-largest country by area and the seventh most popul ous. Its capital is Brasília, and its most popul ous city is São Paulo.

Chunk 2:
The federation is composed of the union of the 26 states and the Federal District. It is the only country in the Americas to have Portugue se as an official language.[11][12]
```

---
## Code Usage Guide
### Start - install packages, dependencies
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" # enter this line if uv is not installed
uv python list
uv python install 3.10.15
uv init --python=3.10.15
uv sync
```

### Data Preparation
./data/corpus.txt --> Text data that needs to be chunked. Fill the file contents with whatever you want.
./data/chunked_corpus.txt --> Chunking result of "./data/corpus.txt". 

### One Click Execution - Execute python file main.py with default setting (Assume text is mainly in korean).
execute batch file in command prompt
```
chunking.bat
```


### User Customized Command
You can choose splitter depend on text language and can adjust threshold for clustering strength.
Please enter command 
```
python main.py --splitter_name [SPLITTER NAME] --threshold [THRESHOLD]
# [SPLITTER NAME]: 'spacy_senter'' is for english, 'kss' is for korean. 
# [THRESHOLD]: Choose similarity threshold for adjacent sentences clustering. value is between 0~1. Higher threshold lead more scattered clusters.
```



