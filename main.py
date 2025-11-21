import utils
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parameter option of document chunking")
    parser.add_argument('--splitter_name', type=str, default="kss", help="Choose splitter. 'spacy_senter'' is for english, 'kss' is for korean. Please refer utils/TextProcessor().")
    parser.add_argument('--vectorizer_name', type=str, default="sentence_transformer_all-MiniLM-L6-v2", help="Choose vectorizer. Please refer utils/TextProcessor().")
    parser.add_argument('--threshold', type=float, default=0.5, help="Choose similarity threshold for adjacent sentences clustering. value is between 0~1")
    args = parser.parse_args()

    final_texts, clusters_lens = utils.adjacent_sentence_clustering(text_path="./data/corpus.txt", splitter_name=args.splitter_name, 
                                                                    vectorizer_name=args.vectorizer_name, threshold=args.threshold)
    utils.save_chunks(chunks=final_texts, save_path="./data/chunked_corpus.txt")