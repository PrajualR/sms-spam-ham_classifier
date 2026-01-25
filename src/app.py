import streamlit as st
import pandas as pd
import time
import os
import pickle
# from gensim.models import Word2Vec
import nltk
# from nltk.tokenize import word_tokenize
import plotly.express as px
import plotly.graph_objects as go

from data_preprocessing import preprocess_text, download_nltk_resources
# from feature_engineering import get_avg_word2vec
download_nltk_resources()

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set page configuration
st.set_page_config(
    page_title="SMS Spam-Ham Detection",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Function to load models
@st.cache_resource
def load_models():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(base_path, ".."))
        model_path = os.path.join(project_root, "models", "spam_classifier.pkl")
        # w2v_path = os.path.join(project_root, "models", "word2vec.model")
        tfidf_path = os.path.join(project_root, "models", "tfidf_vectorizer.pkl")

        # Load the models
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # w2v_model = Word2Vec.load(w2v_path)
        with open(tfidf_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        # return model, w2v_model
        return model, tfidf_vectorizer

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Make sure to run model_training.py first!")
        return None, None

# Function to make predictions
def predict_spam(text, model, tfidf_vectorizer):
    # if model is None or w2v_model is None:
    if model is None or tfidf_vectorizer is None:
        st.error("Models not loaded. Run model_training.py first!")
        return None
    # Preprocess the text
    cleaned_text = preprocess_text(text)
    # # Tokenize the cleaned text
    # tokens = word_tokenize(cleaned_text)
    # Convert to word embeddings
    # vector = get_avg_word2vec(tokens, w2v_model, w2v_model.vector_size)
    vector = tfidf_vectorizer.transform([cleaned_text])

    # # Reshape for model input (single sample)
    # vector = vector.reshape(1, -1)
    # Make prediction
    # prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    spam_prob = probabilities[1]

    # Apply custom threshold
    if spam_prob >= 0.25:
        prediction = 1   # SPAM
    else:
        prediction = 0   # HAM

    result = {
        "is_spam": bool(prediction),
        "prediction": "SPAM" if prediction == 1 else "HAM (Not Spam)",
        "probability_spam": float(probabilities[1]),
        "probability_ham": float(probabilities[0])
    }

    return result


def main():
    # model, w2v_model = load_models()
    model, tfidf_vectorizer = load_models()
    # #Sidebar with app info
    # with st.sidebar:
    #     st.title("📱 SMS Spam Classifier")
    #     st.markdown("### About")
    #     st.info(
    #         "This application uses machine learning to predict whether an SMS "
    #         "message is spam or ham (not spam)."
    #     )

    # Main content
    st.title("SMS Spam/Ham Detector")
    # Create tabs
    tab1, tab2 = st.tabs(["Single Message", "Batch Analysis"])

    # Tab 1: Single message analysis
    with tab1:
        st.markdown("### Check if your message is spam or ham")

        # Single message text input
        message = st.text_area(
            "Enter a message",
            height=120,
            placeholder="Type or paste your SMS message here..."
        )

        if st.button("Check Message", type="primary", width="stretch"):
            if not message:
                st.warning("Please enter a message to analyze.")
            else:
                with st.spinner("Analyzing..."):
                    # Add slight delay for better user experience
                    time.sleep(0.5)
                    # result = predict_spam(message, model, w2v_model)
                    result = predict_spam(message, model, tfidf_vectorizer)

                if result:
                    # Display result
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        # Simple result display
                        result_color = "#FF5151" if result["is_spam"] else "#52D273"
                        st.markdown(
                            f"""
                            <div style="padding: 20px; border-radius: 10px; background-color: {result_color}20; 
                                        border-left: 5px solid {result_color};">
                                <h2 style="color: {result_color};">{result["prediction"]}</h2>
                                <h3>Confidence: {result["probability_spam"] * 100:.1f}% spam probability</h3>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with col2:
                        # Gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=result["probability_spam"] * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Spam Probability (%)", 'font': {'size': 18}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1},
                                'bar': {'color': "rgba(0, 0, 0, 0)"},
                                'steps': [
                                    {'range': [0, 30], 'color': "#52D273"},
                                    {'range': [30, 70], 'color': "#FFDD53"},
                                    {'range': [70, 100], 'color': "#FF5151"},
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': result["probability_spam"] * 100
                                }
                            }
                        ))

                        fig.update_layout(
                            height=250,
                            margin=dict(l=20, r=20, t=50, b=20),
                        )

                        st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Batch analysis
    with tab2:
        st.markdown("### Analyze Multiple Messages")

        # File upload instructions
        st.info("Upload a CSV file containing messages to analyze in bulk.")

        # File uploader for CSV
        uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                # Read the file
                df = pd.read_csv(uploaded_file)

                # Display the first few rows
                st.subheader("Preview of uploaded data")
                st.dataframe(df.head())

                # Allow user to select column
                if len(df.columns) > 0:
                    message_col = st.selectbox(
                        "Select the column containing the messages:",
                        df.columns.tolist()
                    )

                    if st.button("Analyze All Messages", type="primary"):
                        # Process messages in batches
                        results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for i, message in enumerate(df[message_col]):
                            if isinstance(message, str):
                                # result = predict_spam(message, model, w2v_model)
                                result = predict_spam(message, model, tfidf_vectorizer)
                                if result:
                                    results.append({
                                        'message': message,
                                        'prediction': result['prediction'],
                                        'is_spam': result['is_spam'],
                                        'spam_probability': result['probability_spam']
                                    })

                            # Update progress
                            progress = (i + 1) / len(df[message_col])
                            progress_bar.progress(progress)
                            status_text.text(f"Processed {i + 1} of {len(df[message_col])} messages...")

                        # Create results dataframe
                        results_df = pd.DataFrame(results)

                        if not results_df.empty:
                            # Display results
                            st.subheader("Analysis Results")
                            st.dataframe(results_df)

                            # Summary visualization
                            st.subheader("Summary")

                            # Pie chart of predictions
                            prediction_counts = results_df['is_spam'].value_counts().reset_index()
                            prediction_counts.columns = ['Is Spam', 'Count']
                            prediction_counts['Category'] = prediction_counts['Is Spam'].map(
                                {True: 'Spam', False: 'Ham'})

                            fig = px.pie(
                                prediction_counts,
                                values='Count',
                                names='Category',
                                title='Message Classification Results',
                                color='Category',
                                color_discrete_map={'Ham': '#1f77b4', 'Spam': '#ff7f0e'}
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Add download button for results
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download Results as CSV",
                                csv,
                                "spam_analysis_results.csv",
                                "text/csv",
                                key='download-csv'
                            )
                        else:
                            st.warning("No valid messages found in the selected column.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")


if __name__ == "__main__":
    main()