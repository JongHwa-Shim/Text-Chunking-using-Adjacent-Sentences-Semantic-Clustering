import os
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

class TextProcessor():
    def __init__(self, splitter_name = "spacy", vectorizer_name='tok2vec'):
        """
        <Splitter>
        spacy_senter: spacy Tagger v2 based sentence splitter. Can only process english.
        kss (Korean Sencetence Spliter): Works in Korean-dominant environments.
        
        <Vectorizer>
        tok2vec: HashEmbedCNN based model. Fastest but only can process english. Doesn't have sequence length limitation, but recommand to organize input text to singel sentence or paragraph.
                Output dimension is 96.
        sentence_transformer_all-mpnet-base-v2: mpnet based model. Multilingual. Provides the best quality. Max sequence length is 384. Output dimension is 768. 
        sentence_transformer_all-MiniLM-L6-v2: MiniLM based model. Multilingual. 5 times faster than mpnet-base-v2 and still offers good quality. 
                                                Max sequence length is 256. Output dimension is 384.
        """
        self.splitter_name = splitter_name
        self.vectorizer_name = vectorizer_name

        # Initialize splitter
        if self.splitter_name == "spacy_senter":
            self.splitter = spacy.load('en_core_web_sm')
        elif self.splitter_name == "kss":
            import kss
            self.splitter = kss
        else:
            raise(ValueError)
        
        # Initialize vectorizer
        if self.vectorizer_name == "tok2vec":
            self.vectorizer = spacy.load("en_core_web_sm")
        elif self.vectorizer_name == "sentence_transformer_all-mpnet-base-v2":
            self.vectorizer = SentenceTransformer("all-mpnet-base-v2")
        elif self.vectorizer_name == "sentence_transformer_all-MiniLM-L6-v2":
            self.vectorizer = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            raise(ValueError)

    def split_text(self, text):
        if self.splitter_name == "spacy_senter":
            doc = self.splitter(text)
            sents = list(doc.sents)
            sents = [sent.lemma_ for sent in sents]
        elif self.splitter_name == "kss":
            sents = self.splitter.split_sentences(text)
        else:
            raise(ValueError)
        
        return sents

    
    def vectorize_text(self, sent):
        if self.vectorizer_name == "tok2vec":
            sent = self.vectorizer(sent)
            sent_vector = sent.vector / sent.vector_norm
        elif self.vectorizer_name == "sentence_transformer_all-mpnet-base-v2":
            sent_vector = self.vectorizer.encode(sent, normalize_embeddings=True)
        elif self.vectorizer_name == "sentence_transformer_all-MiniLM-L6-v2":
            sent_vector = self.vectorizer.encode(sent, normalize_embeddings=True)

        return sent_vector # Normalized vector (unit vector)
    
    # Wrapper of vectorize_text
    def vectorize_sents(self, sents):
        vecs = np.stack([self.vectorize_text(sent) for sent in sents])
        
        return vecs
    
    def process(self, text):
        sents = self.split_text(text)
        vecs = self.vectorize_sents(sents)

        return sents, vecs

def cluster_text(sents, vecs, threshold):
    clusters = [[0]]
    for i in range(1, len(sents)):
        if np.dot(vecs[i], vecs[i-1]) < threshold:
            clusters.append([])
        clusters[-1].append(i)
    
    return clusters

def postprocess_text(text):
    # Add your text cleaning postprocess here
    return text

def read_text_file(text_path):
    file = open(text_path, 'r')
    line_list = file.readlines()
    text = ' '.join(line_list)
    # Can add some text preprocessing procedure here
    return text

def save_chunks(chunks: list[str], save_path="./chunked_corpus.txt"):
    with open(save_path, 'w') as f:
        for i, chunk in enumerate(chunks):
            chunk_num = i+1

            chunk_content = "Chunk #{chunk_num}\n" \
            "{chunk_text}\n\n".format(chunk_num=chunk_num, chunk_text=chunk)

            f.write(chunk_content)
        

def adjacent_sentence_clustering(text_path, splitter_name="spacy", vectorizer_name="tok2vec", threshold=0.3, min_cluster_len=0):
    text = read_text_file(text_path)

    # Initialize the clusters lengths list and final texts list
    clusters_lens = []
    final_texts = []

    # Set parameters
    text_processor = TextProcessor(splitter_name=splitter_name, vectorizer_name=vectorizer_name)
    threshold = threshold
    if text_processor.vectorizer_name == "tok2vec":
        max_cluster_len = 500
    elif text_processor.vectorizer_name == "sentence_transformer_all-mpnet-base-v2":
        max_cluster_len = 1000
    elif text_processor.vectorizer_name == "sentence_transformer_all-MiniLM-L6-v2":
        max_cluster_len = 700

    # Process the chunk
    sents, vecs = text_processor.process(text)

    # Cluster the sentences
    clusters = cluster_text(sents, vecs, threshold)

    for cluster in clusters:
        cluster_txt = postprocess_text(' '.join([sents[i] for i in cluster]))
        cluster_len = len(cluster_txt)
        
        # Check if the cluster is too short
        if cluster_len < min_cluster_len:
            continue
        
        # Check if the cluster is too long
        elif cluster_len > max_cluster_len:
            threshold = 0.6
            sents_div, vecs_div = text_processor.process(cluster_txt)
            reclusters = cluster_text(sents_div, vecs_div, threshold)
            
            for subcluster in reclusters:
                div_txt = postprocess_text(' '.join([sents_div[i] for i in subcluster]))
                div_len = len(div_txt)
                
                if div_len < min_cluster_len or div_len > max_cluster_len:
                    continue
                
                clusters_lens.append(div_len)
                final_texts.append(div_txt)
                
        else:
            clusters_lens.append(cluster_len)
            final_texts.append(cluster_txt)
            
    return final_texts, clusters_lens