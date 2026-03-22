import pandas as pd
import ftfy
from tqdm import tqdm

folder=r"C:\Users\wesle\Downloads\NewspapersMalayaTribune1919to1923.csv"

def fix_encoding(text):
	"""Fix mojibake and encoding glitches using ftfy."""
	if not text or not isinstance(text, str):
		return text
	return ftfy.fix_text(text)

def process(data=None):
	if data is None:
		data = folder
	df = pd.read_csv(data)

	# If the column exists, drop rows containing the phrase (case-insensitive)
	if 'article_title' in df.columns:
		mask = ~df['article_title'].astype(str).str.contains('Advertisements', case=False, na=False)
		df = df.loc[mask]
		print(f"Filtered to {len(df)} documents (removed advertisements)")
	
	# Apply only fast encoding fix to article content
	if 'article_text_1st50words' in df.columns:
		texts = df['article_text_1st50words'].tolist()
		df['article_text_1st50words'] = [fix_encoding(text) for text in tqdm(texts, desc="Fixing encoding")]
	
	return df

