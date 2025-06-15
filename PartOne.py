#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import math
import nltk
import spacy
from pathlib import Path
import pandas as pd
from collections import Counter


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    num_words = len(words)
    num_sentences = len(sentences)
    num_syllables = sum(count_subjectsyl(word, d) for word in words)

    if num_words == 0 or num_sentences == 0:
        return 0.0
    
    # Calculate the Flesch-Kincaid Grade Level
    # 0.39 * (total words / total sentences) + 11.8 * (total syllables / total words) - 15.59
    return 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59

def count_subjectsyl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    if word.lower() in d:
        return len(d[word.lower()])
    else:
        vowels = "aeiou"
        prev_char_is_vowel = False
        count = 0
        for char in word.lower():
            if char in vowels:
                if not prev_char_is_vowel:
                    count += 1
                prev_char_is_vowel = True
            else:
                prev_char_is_vowel = False
    return count
    


def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    novels = []
    for file in path.glob("*.txt"):
        with open(file, 'r', encoding='utf-8') as f:
            name = file.stem.split("-")
            text = f.read()
            title = name[0].strip().replace("_", " ")
            author = name[1].strip() if len(name) > 1 else ""
            year = name[2].strip() if len(name) > 2 else ""
            novels.append({
                "text": text,
                "title": title,
                "author": author,
                "year": year
            })
    df = pd.DataFrame(novels)
    df = df.sort_values(by="year").reset_index(drop=True)
    return df


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    if not store_path.exists():
        store_path.mkdir(parents=True)
    
    results = []
    for i, row in df.iterrows():
        doc = nlp(row["text"])
        results.append({
            "title": row["title"],
            "text": row["text"],
            "author": row["author"],
            "year": row["year"],
            "parsed": doc
        })
    
    parsed_df = pd.DataFrame(results)
    parsed_df.to_pickle(store_path / out_name)
    return parsed_df


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    types = set(tokens)
    if len(tokens) == 0:
        return 0.0
    return len(types) / len(tokens) if len(tokens) > 0 else 0.0


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subjects_verb_pairs = []
    subject_counts = Counter()
    verb_counts = Counter()
    total_pairs = 0

    for token in doc:
        if token.pos_ == 'VERB':
            verb_lemma = token.lemma_
            for child in token.children:
                if child.dep_ in ('nsubj', 'nsubjpass'):
                    subject = child.text
                    subjects_verb_pairs.append((subject, verb_lemma))
                    subject_counts[subject] += 1
                    verb_counts[verb_lemma] += 1
                    total_pairs += 1


    subjects_by_verb_count = Counter(
        subject for subject, verb in subjects_verb_pairs if verb == target_verb
    )

    pmi_results = []
    for subject, count_subjects_verbs in subjects_by_verb_count.items():
        count_subjects = subject_counts[subject]
        count_verbs = verb_counts[target_verb]
        if count_subjects_verbs == 0 or count_subjects == 0 or count_verbs == 0:
            continue
        probability_subjects_verbs = count_subjects_verbs / total_pairs
        probability_subjects = count_subjects / total_pairs
        probability_verbs = count_verbs / total_pairs
        pmi = math.log2(probability_subjects_verbs / (probability_subjects * probability_verbs))
        pmi_results.append((subject, pmi))

    return sorted(pmi_results, key=lambda x: x[1], reverse=True)[:10]

def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subjects = []
    for token in doc:
        if token.lemma_ == verb and token.pos_ == 'VERB':
            for child in token.children:
                if child.dep_ in ('nsubj', 'nsubjpass'):
                    subjects.append(child.text)

    subject_counts = Counter(subjects)
    return subject_counts.most_common(10)



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    adjectives = []
    for token in doc:
        if token.pos_ == 'ADJ':
            adjectives.append(token.text)
    adjective_counts = Counter(adjectives)
    return adjective_counts.most_common(10)

if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    nltk.download("cmudict")
    parse(df)
    print(df.head())
    print(get_ttrs(df))
    print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    for i, row in df.iterrows():
        print(row["title"])
        print(adjective_counts(row["parsed"]))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")

