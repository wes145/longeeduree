import pandas as pd
from bertopic import BERTopic

def fit_bertopic(df, text_column=None, min_topic_size=10, language='english'):
	"""Fit BERTopic model on preprocessed documents."""
	# Auto-detect text column if not specified
	if text_column is None:
		candidates = ['article_text_1st50words', 'article_text', 'text', 'content', 'body']
		text_column = next((c for c in candidates if c in df.columns), None)
		if text_column is None:
			print(f"Error: Could not find text column. Available columns: {df.columns.tolist()}")
			return None, None, None, None
	
	if text_column not in df.columns or len(df) == 0:
		print(f"Error: Column '{text_column}' not found or dataframe is empty.")
		return None, None, None, None
	
	documents = df[text_column].astype(str).tolist()
	
	# Initialize and fit BERTopic
	topic_model = BERTopic(
		language=language,
		min_topic_size=min_topic_size
	)
	topics, probs = topic_model.fit_transform(documents)
	
	return topic_model, topics, probs, documents

def generate_overview(topic_model, topics, df, documents):
	"""Generate overview of BERTopic results."""
	if topic_model is None:
		print("No topic model available.")
		return None
	
	overview = {
		'num_topics': len(set(topics)) - 1,  # -1 for noise (-1 label)
		'num_documents': len(documents),
		'topic_info': topic_model.get_topic_info(),
		'document_topic_distribution': pd.DataFrame({
			'document_id': range(len(documents)),
			'topic': topics,
			'document': documents[:100]  # first 100 for preview
		})
	}
	
	# Add sample documents per topic
	overview['sample_docs_per_topic'] = {}
	for topic_id in set(topics):
		if topic_id != -1:  # skip noise
			indices = [i for i, t in enumerate(topics) if t == topic_id][:3]
			overview['sample_docs_per_topic'][topic_id] = [documents[i] for i in indices]
	
	return overview

def process(preprocessed_df):
	"""Run BERTopic on preprocessed dataframe and return results."""
	topic_model, topics, probs, documents = fit_bertopic(preprocessed_df, min_topic_size=10)
	
	if topic_model is None:
		return None
	
	overview = generate_overview(topic_model, topics, preprocessed_df, documents)
	
	# Print summary
	print("\n=== BERTopic Results Overview ===")
	print(f"Total documents: {overview['num_documents']}")
	print(f"Number of topics: {overview['num_topics']}")
	
	return {
		'model': topic_model,
		'topics': topics,
		'probabilities': probs,
		'overview': overview
	}
	return {
		'model': topic_model,
		'topics': topics,
		'probabilities': probs,
		'overview': overview
	}
