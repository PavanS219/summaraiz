import nltk
import graphviz
import base64
from io import BytesIO
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import streamlit as st
import spacy
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
import os
import sys

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
download_nltk_resources()

# Load spaCy model - Alternative approach for Streamlit
@st.cache_resource
def load_spacy_model():
    try:
        # First try loading directly
        return spacy.load('en_core_web_sm')
    except:
        # If that fails, try downloading
        st.info("First-time setup: Downloading language model...")
        # Streamlit-recommended approach
        try:
            # Method 1: Python executable approach
            os.system(f"{sys.executable} -m spacy download en_core_web_sm")
            return spacy.load('en_core_web_sm')
        except:
            try:
                # Method 2: PIP install directly
                os.system(f"{sys.executable} -m pip install en-core-web-sm")
                return spacy.load('en_core_web_sm')
            except:
                # Method 3: Direct import approach
                try:
                    from spacy.cli import download
                    download('en_core_web_sm')
                    return spacy.load('en_core_web_sm')
                except Exception as e:
                    st.error(f"Failed to load spaCy model: {str(e)}")
                    # Fallback to minimal processing
                    return spacy.blank("en")

# Function to get semantic meaning and importance of words
def get_key_entities_and_concepts(text, nlp, max_entities=15):
    """Extract named entities, noun phrases, and important concepts from text."""
    doc = nlp(text)
    
    # Extract named entities (people, places, organizations, etc.)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Extract noun chunks (meaningful phrases)
    noun_chunks = [chunk.text for chunk in doc.noun_chunks 
                   if len(chunk.text.split()) <= 3]  # Keep phrases relatively short
    
    # Get lemmatized nouns and proper nouns
    pos_words = []
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and len(token.text) > 2:
            pos_words.append(token.lemma_)
    
    # Combine all potential concepts and count frequencies
    all_concepts = []
    all_concepts.extend([e[0].lower() for e in entities])
    all_concepts.extend([chunk.lower() for chunk in noun_chunks])
    all_concepts.extend(pos_words)
    
    # Count frequencies and get the most common concepts
    concept_freq = defaultdict(int)
    for concept in all_concepts:
        concept_freq[concept] += 1
    
    # Sort by frequency and return top concepts
    sorted_concepts = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)
    return [concept for concept, freq in sorted_concepts[:max_entities]]

# Get WordNet POS tag from NLTK POS tag
def get_wordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

# Extract keywords with semantic meaning
def extract_semantic_keywords(text, nlp, num_keywords=10):
    """Extract meaningful keywords using spaCy and TF-IDF."""
    # Clean text of special characters
    clean_text = re.sub(r'[^\w\s]', '', text)
    
    # Get semantically meaningful entities and concepts
    semantic_keywords = get_key_entities_and_concepts(text, nlp)
    
    # If we don't have enough semantic keywords, supplement with TF-IDF
    if len(semantic_keywords) < num_keywords:
        stop_words = set(stopwords.words('english'))
        
        # Tokenize and lemmatize
        lemmatizer = WordNetLemmatizer()
        words = nltk.word_tokenize(clean_text.lower())
        tagged_words = nltk.pos_tag(words)
        
        lemmatized_words = []
        for word, tag in tagged_words:
            if word.isalnum() and word not in stop_words and len(word) > 2:
                wordnet_pos = get_wordnet_pos(tag)
                lemmatized_words.append(lemmatizer.lemmatize(word, wordnet_pos))
        
        lemmatized_text = ' '.join(lemmatized_words)
        
        # Use TF-IDF for remaining keywords
        vectorizer = TfidfVectorizer(max_features=num_keywords, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform([lemmatized_text])
            tfidf_keywords = vectorizer.get_feature_names_out()
            
            # Add TF-IDF keywords that aren't already in semantic keywords
            for keyword in tfidf_keywords:
                if keyword not in semantic_keywords and len(semantic_keywords) < num_keywords:
                    semantic_keywords.append(keyword)
        except:
            pass  # If TF-IDF fails, just use what we have
    
    # Limit to the requested number of keywords
    return semantic_keywords[:num_keywords]

# Simplified fallback approach if spaCy model not available
def extract_fallback_keywords(text, num_keywords=10):
    """Extract keywords without using spaCy if model fails to load."""
    # Clean text
    clean_text = re.sub(r'[^\w\s]', '', text)
    
    # Use TF-IDF for keywords
    stop_words = set(stopwords.words('english'))
    
    # Tokenize and filter
    words = nltk.word_tokenize(clean_text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]
    filtered_text = ' '.join(filtered_words)
    
    # Use TF-IDF
    vectorizer = TfidfVectorizer(max_features=num_keywords, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([filtered_text])
        keywords = vectorizer.get_feature_names_out()
        return list(keywords)
    except:
        # If TF-IDF fails, return most common words
        from collections import Counter
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(num_keywords)]

# Function to determine keyword relationships and build mindmap structure
def build_semantic_mindmap_structure(keywords, text, nlp, max_main_topics=3):
    """Create a hierarchical structure for the mindmap based on semantic relationships."""
    # If no keywords, create a default structure
    if not keywords:
        return {"Mindmap": []}, "Mindmap"
    
    # Check if we have a full spaCy model with vectors
    has_vectors = hasattr(nlp, 'has_pipe') and nlp.has_pipe('tok2vec')
    
    if has_vectors:
        # Process the text with spaCy to get embeddings
        doc = nlp(text)
        
        # Find the most central concept (the one with highest average similarity to others)
        keyword_docs = [nlp(keyword) for keyword in keywords]
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(keywords), len(keywords)))
        for i, doc1 in enumerate(keyword_docs):
            for j, doc2 in enumerate(keyword_docs):
                if i != j:
                    similarity_matrix[i, j] = doc1.similarity(doc2)
        
        # Get the keyword with highest average similarity to others
        avg_similarities = np.mean(similarity_matrix, axis=1)
        root_index = np.argmax(avg_similarities)
        root = keywords[root_index]
        
        # Remove root from keywords
        remaining_keywords = keywords.copy()
        remaining_keywords.pop(root_index)
        remaining_docs = keyword_docs.copy()
        remaining_docs.pop(root_index)
        
        # Calculate similarities to root
        root_similarities = []
        root_doc = nlp(root)
        for i, kw_doc in enumerate(remaining_docs):
            root_similarities.append((remaining_keywords[i], kw_doc.similarity(root_doc)))
        
        # Sort by similarity to root
        root_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Select main topics (most similar to root but limited in number)
        main_topics = [kw for kw, _ in root_similarities[:max_main_topics]]
        sub_topics = [kw for kw, _ in root_similarities[max_main_topics:]]
        
        # Build hierarchy
        hierarchy = defaultdict(list)
        
        # Add main topics as children of root
        for topic in main_topics:
            hierarchy[root].append(topic)
        
        # Distribute sub-topics among main topics based on similarity
        if main_topics and sub_topics:
            main_topic_docs = [nlp(topic) for topic in main_topics]
            
            for sub in sub_topics:
                sub_doc = nlp(sub)
                similarities = [sub_doc.similarity(topic_doc) for topic_doc in main_topic_docs]
                best_match_idx = np.argmax(similarities)
                best_match = main_topics[best_match_idx]
                hierarchy[best_match].append(sub)
    else:
        # Fallback to a simpler approach if no vectors
        # Just choose the first keyword as root and distribute others
        if keywords:
            root = keywords[0]
            remaining = keywords[1:]
            
            # Simple distribution - every n keywords go to a main topic
            hierarchy = defaultdict(list)
            
            # If we have at least max_main_topics remaining keywords
            if len(remaining) >= max_main_topics:
                main_topics = remaining[:max_main_topics]
                sub_topics = remaining[max_main_topics:]
                
                # Add main topics as children of root
                for topic in main_topics:
                    hierarchy[root].append(topic)
                
                # Distribute subtopics
                for i, sub in enumerate(sub_topics):
                    main_idx = i % len(main_topics)
                    hierarchy[main_topics[main_idx]].append(sub)
            else:
                # All remaining keywords are main topics
                for topic in remaining:
                    hierarchy[root].append(topic)
    
    return hierarchy, root

# Function to generate mindmap from text and return base64 encoded image
def generate_semantic_mindmap(text, theme="dark"):
    """Generate a semantically meaningful mindmap visualization from text."""
    # Get spaCy model
    nlp = load_spacy_model()
    
    # Check if we have a full spaCy model with named entities and vectors
    has_ner = hasattr(nlp, 'has_pipe') and nlp.has_pipe('ner')
    
    # Extract keywords - use appropriate method based on model capabilities
    if has_ner:
        keywords = extract_semantic_keywords(text, nlp, num_keywords=12)
    else:
        # Use fallback if we don't have a full model
        keywords = extract_fallback_keywords(text, num_keywords=12)
    
    # Build semantic mindmap structure
    hierarchy, root = build_semantic_mindmap_structure(keywords, text, nlp)
    
    # Configure graph based on theme
    if theme == "dark":
        bgcolor = "#121212"
        fontcolor = "white"
        root_color = "#6A5ACD"  # Slate blue
        main_topic_color = "#DAA520"  # Goldenrod
        subtopic_color = "#2E8B57"  # Sea green
        edge_color = "#FFFFFF"
    else:
        bgcolor = "#FFFFFF"
        fontcolor = "black"
        root_color = "#4B0082"  # Indigo
        main_topic_color = "#FF8C00"  # Dark orange
        subtopic_color = "#228B22"  # Forest green
        edge_color = "#000000"
    
    # Create the graph
    graph = graphviz.Digraph(format='png')
    graph.attr(bgcolor=bgcolor, fontcolor=fontcolor, rankdir="TB", 
               splines="curved", concentrate="true")
    
    # Format node labels (capitalize first letter, handle multiple words)
    def format_label(text):
        words = text.split()
        formatted_words = [word.capitalize() for word in words]
        return " ".join(formatted_words)
    
    # Add nodes and edges
    for parent, children in hierarchy.items():
        # Format parent node
        parent_label = format_label(parent)
        
        if parent == root:
            graph.node(parent, parent_label, shape='box', style='filled,rounded', 
                      fillcolor=root_color, fontcolor='white', fontsize="16", penwidth="2")
        else:
            graph.node(parent, parent_label, shape='box', style='filled,rounded', 
                      fillcolor=main_topic_color, fontcolor='white')
            
        for child in children:
            # Format child node
            child_label = format_label(child)
            
            graph.node(child, child_label, shape='box', style='filled,rounded', 
                      fillcolor=subtopic_color, fontcolor='white')
            graph.edge(parent, child, color=edge_color, penwidth="1.5")
    
    # Render the graph to a PNG image
    png_data = graph.pipe(format='png')
    
    # Encode to base64 for display in HTML
    encoded = base64.b64encode(png_data).decode('utf-8')
    
    return encoded

# UI section for the mindmap
def add_semantic_mindmap_section(summary_text, dark_mode=True, timestamp=None):
    """Add the semantic mindmap UI section to the Streamlit app."""
    if summary_text:
        st.markdown("---")
        st.subheader("Key Concepts Mindmap")
        st.write("Visual representation of the key concepts in the summary.")
        
        # Use the current theme state
        theme = "dark" if dark_mode else "light"
        
        with st.spinner("Generating mindmap..."):
            try:
                encoded_image = generate_semantic_mindmap(summary_text, theme)
                st.markdown(f'<img src="data:image/png;base64,{encoded_image}" width="100%">', 
                           unsafe_allow_html=True)
                
                # Add download button for the mindmap image
                buffered = BytesIO(base64.b64decode(encoded_image))
                
                # Generate a timestamp-based filename if timestamp is provided
                file_name = f"mindmap_{timestamp}.png" if timestamp else "mindmap.png"
                
                st.download_button(
                    label="Download Mindmap",
                    data=buffered.getvalue(),
                    file_name=file_name,
                    mime="image/png",
                    key="mindmap_download"
                )
            except Exception as e:
                st.error(f"Failed to generate mindmap: {str(e)}")
                st.info("The mindmap generation requires text with sufficient content to identify key concepts.")
    else:
        st.info("Generate a summary first to see the mindmap visualization.")

# Main app function
def main():
    st.title("Semantic Mindmap Generator")
    st.write("Enter text to generate a mindmap of key concepts and their relationships.")
    
    # Text input
    input_text = st.text_area("Enter your text (minimum 100 characters recommended):", 
                            height=200,
                            placeholder="Paste your text here...")
    
    # Theme selection
    dark_mode = st.checkbox("Dark Theme", value=True)
    
    if st.button("Generate Mindmap"):
        if input_text and len(input_text) >= 50:  # Minimum length check
            with st.spinner("Processing text..."):
                add_semantic_mindmap_section(input_text, dark_mode)
        else:
            st.warning("Please enter more text for better mindmap generation (at least 50 characters).")

if __name__ == "__main__":
    main()
