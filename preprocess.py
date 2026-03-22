import csv
import pandas as pd
import spacy
import symspellpy
import ftfy
import os
from tqdm import tqdm

folder=r"C:\Users\wesle\Downloads\NewspapersMalayaTribune1919to1923.csv"

# Load spacy model (downloads if needed)
try:
	nlp = spacy.load("en_core_web_sm")
except OSError:
	print("Downloading spacy model...")
	os.system("python -m spacy download en_core_web_sm")
	nlp = spacy.load("en_core_web_sm")

# Initialize symspellpy for spelling correction
sym_spell = symspellpy.SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = os.path.join(os.path.dirname(__file__), "frequency_bigramedEN_80_0.txt")
# If dictionary not found locally, symspellpy will fall back or you can download it
if os.path.exists(dictionary_path):
	sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def fix_encoding(text):
	"""Fix mojibake and encoding glitches using ftfy."""
	if not text or not isinstance(text, str):
		return text
	return ftfy.fix_text(text)

def normalize_spelling(text, max_edit_distance=2):
	"""Correct spelling errors using symspellpy."""
	if not text or not isinstance(text, str):
		return text
	suggestions = sym_spell.lookup_compound(text, max_edit_distance=max_edit_distance)
	return suggestions[0].term if suggestions else text

def tokenize_and_lemmatize(text):
	"""Tokenize and lemmatize text using spacy."""
	if not text or not isinstance(text, str):
		return []
	doc = nlp(text.lower())
	return [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]

def preprocess_text(text):
	"""Full pipeline: encoding fix -> spelling -> tokenization -> lemmatization."""
	fixed = fix_encoding(text)
	normalized = normalize_spelling(fixed)
	tokens = tokenize_and_lemmatize(normalized)
	return tokens

def process(data=None):
	if data is None:
		data = folder
	df = pd.read_csv(data)

	# If the column exists, drop rows containing the phrase (case-insensitive)
	if 'article_title' in df.columns:
		mask = ~df['article_title'].astype(str).str.contains('Advertisements', case=False, na=False)
		df = df.loc[mask]
		print(f"Filtered to {len(df)} documents (removed advertisements)")
	
	# Apply preprocessing to article content (use correct column name)
	if 'article_text_1st50words' in df.columns:
		texts = df['article_text_1st50words'].tolist()
		df['tokens'] = [preprocess_text(text) for text in tqdm(texts, desc="Preprocessing text")]
	
	return df

