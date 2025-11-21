# Text Chunking using Adjacent Sentences Semantic Clustering (인접 문장 의미론적 군집화를 사용한 텍스트 청킹)

- 인접 문장간의 유사도를 기반으로 군집화 → 각 군집을 chunk화
- 유사도 계산을 위한 벡터화 모델로 Sentence Transformer를 사용하여 **폐쇄망에서도 구현가능**
- 유사도 임계값 또는 chunk 크기 최소&최댓값을 조절하여 임의의 **chunking 환경 조절 가능**
- **Pros:** 각 chunk의 콘텐츠 일관성, 유연한 chunk 크기, threshold 등 매개변수 조정 가능, 순서 보존
- **Cons:** 방법이 다소 복잡, 처리시간 증가, chunk 크기의 균일성이 중요하다면 효과적이지 못함, 텍스트 컨텐츠의 구성이 매우 복잡하다면(문장, 단어 파편, 형식 부호 등이 뒤섞인) sentence transformer의 벡터화 성능이 하락할 수 있음

## Chunking Comparison
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
### Data Pipeline
./data/corpus.txt --> Text data that needs to be chunked. Please the file contents with whatever you want.
./data/chunked_corpus.txt --> Chunking result of "./data/corpus.txt". 

### Start - python installation
Please install python version 3.10.

### Start - install packages, dependencies
Please execute 
```
installation.bat
```
in the command prompt (python 3.10 environment).

### One Click Execution - Execute python file main.py with default setting (Assume text is mainly in korean, use sentence transformer vectorizer).
Please execute 
```
chunking.bat
```
in command prompt (python 3.10 environment).

### User Customized Command
Please enter command 
```
python main.py --splitter_name [SPLITTER NAME] --vectorizer_name [VECTORIZER NAME] --threshold [THRESHOLD]
```



