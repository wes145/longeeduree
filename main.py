import sys
import pandas as pd
from preprocess import process as preprocess
from process import process as bertopic_process

def main():
	"""Main pipeline: preprocess -> BERTopic analysis."""
	
	print("Step 1: Loading and preprocessing data...")
	try:
		preprocessed_df = preprocess()
		print(f"✓ Preprocessed {len(preprocessed_df)} documents (after filtering advertisements)")
		print(f"  Columns: {preprocessed_df.columns.tolist()}")
	except Exception as e:
		print(f"✗ Preprocessing failed: {e}")
		sys.exit(1)
	
	if len(preprocessed_df) == 0:
		print("✗ No documents to process after filtering.")
		sys.exit(1)
	
	print("\nStep 2: Running BERTopic analysis...")
	try:
		results = bertopic_process(preprocessed_df)
		if results is None:
			print("✗ BERTopic processing failed.")
			sys.exit(1)
	except Exception as e:
		print(f"✗ BERTopic processing failed: {e}")
		sys.exit(1)
	
	print("\n" + "="*60)
	print("FINAL RESULTS")
	print("="*60)
	print(f"Total documents analyzed: {results['overview']['num_documents']}")
	print(f"Topics discovered: {results['overview']['num_topics']}")
	
	# Newspaper and date statistics
	if 'newspaper_title' in preprocessed_df.columns:
		print(f"\nNewspapers: {preprocessed_df['newspaper_title'].nunique()}")
		print(preprocessed_df['newspaper_title'].value_counts().head(5))
	
	if 'issue_date' in preprocessed_df.columns:
		print(f"\nDate range: {preprocessed_df['issue_date'].min()} to {preprocessed_df['issue_date'].max()}")
	
	print("\nTopic Distribution:")
	topic_info = results['overview']['topic_info']
	if len(topic_info) > 0:
		print(topic_info[['Topic', 'Count', 'Name']].head(15).to_string())
	
	print("\nSample documents per topic:")
	for topic_id, docs in list(results['overview']['sample_docs_per_topic'].items())[:5]:
		print(f"\n  Topic {topic_id}:")
		for i, doc in enumerate(docs, 1):
			print(f"    {i}. {doc[:100]}...")
	
	output_path = r"C:\Users\wesle\Downloads\bertopic_results.csv"
	results['overview']['topic_info'].to_csv(output_path, index=False)
	print(f"\n✓ Results saved to {output_path}")
	
	return results

if __name__ == "__main__":
	results = main()
