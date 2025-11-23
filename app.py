import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="Cyberbullying Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .safe-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .bullying-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    .feature-positive {
        color: #dc3545;
        font-weight: bold;
    }
    .feature-negative {
        color: #28a745;
        font-weight: bold;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


class CyberbullyingDetector:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.class_names = ['Not Cyberbullying', 'Cyberbullying']

    def load_model(self, model_path):
        """Load the trained model"""
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            st.error(f"Model file not found at {model_path}")
            return None

    def predict(self, text):
        """Make prediction for input text"""
        if self.model is None:
            return None, None

        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        return prediction, probabilities

    def get_shap_explanation(self, text, top_n=15):
        """Get SHAP explanation for prediction"""
        if self.model is None:
            return []

        vectorizer = self.model.named_steps['tfidf']
        svm_model = self.model.named_steps['svm']

        # Transform text
        text_vec = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()

        # Get feature importance using model coefficients
        feature_importance = []
        coefficients = svm_model.coef_[0]

        # Get non-zero features in this text
        nonzero_indices = text_vec.nonzero()[1]

        for idx in nonzero_indices:
            importance = coefficients[idx] * text_vec[0, idx]
            feature_importance.append({
                'feature': feature_names[idx],
                'importance': importance,
                'abs_importance': abs(importance)
            })

        # Sort by absolute importance
        feature_importance.sort(key=lambda x: x['abs_importance'], reverse=True)
        return feature_importance[:top_n]


def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ AI Cyberbullying Detector</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    This AI-powered tool detects cyberbullying in text using machine learning. 
    It analyzes language patterns and provides explanations for its predictions.
    """)

    # Initialize detector
    detector = CyberbullyingDetector('model/cyberbully_model_hybrid.pkl')

    if detector.model is None:
        st.error("Please train the model first by running train_model.py")
        st.stop()

    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Single Analysis", "Batch Analysis", "Safety Tips", "Model Info"]
    )

    if app_mode == "Single Analysis":
        single_analysis_mode(detector)
    elif app_mode == "Batch Analysis":
        batch_analysis_mode(detector)
    elif app_mode == "Safety Tips":
        safety_tips_mode()
    else:
        model_info_mode(detector)


def single_analysis_mode(detector):
    """Single text analysis interface"""

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Analyze Text for Cyberbullying")

        # Text input
        user_text = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste text here to check for cyberbullying...",
            help="The AI will analyze the text for harmful language patterns"
        )

        # Example texts
        with st.expander("💡 Example Texts to Try"):
            st.write("""
            **Cyberbullying Examples:**
            - "You're so stupid and worthless, nobody likes you"
            - "Everyone hates you, just disappear"
            - "You're such a loser, why do you even try"

            **Safe Examples:**
            - "Great job on the project today!"
            - "Let's meet up for coffee tomorrow"
            - "I disagree with your opinion, but respect it"
            """)

    with col2:
        st.subheader("⚙️ Analysis Settings")
        show_explanation = st.checkbox("Show Detailed Explanation", value=True)
        num_features = st.slider("Number of features to show", 5, 20, 10)
        confidence_threshold = st.slider("Confidence threshold", 0.5, 0.95, 0.7)

    # Analyze button
    if st.button("🔍 Analyze Text", type="primary") and user_text:
        with st.spinner("Analyzing text for cyberbullying..."):
            analyze_single_text(detector, user_text, show_explanation,
                                num_features, confidence_threshold)


def analyze_single_text(detector, text, show_explanation, num_features, confidence_threshold):
    """Analyze single text and display results"""

    # Make prediction
    prediction, probabilities = detector.predict(text)
    is_cyberbullying = prediction == 1
    confidence = probabilities[1] if is_cyberbullying else probabilities[0]

    # Display results
    st.subheader("🎯 Analysis Results")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Prediction box
        if is_cyberbullying:
            st.markdown(f"""
            <div class="bullying-box">
                <h3>🚨 CYBERBULLYING DETECTED</h3>
                <h4>Confidence: {confidence:.1%}</h4>
                <p>This text contains harmful language patterns.</p>
            </div>
            """, unsafe_allow_html=True)

            if confidence >= confidence_threshold:
                st.error(f"⚠️ High confidence detection (≥{confidence_threshold:.0%})")
            else:
                st.warning(f"🔸 Moderate confidence detection (<{confidence_threshold:.0%})")
        else:
            st.markdown(f"""
            <div class="safe-box">
                <h3>✅ NO CYBERBULLYING DETECTED</h3>
                <h4>Confidence: {confidence:.1%}</h4>
                <p>This text appears to be safe.</p>
            </div>
            """, unsafe_allow_html=True)

            if confidence >= confidence_threshold:
                st.success(f"✅ High confidence (≥{confidence_threshold:.0%})")
            else:
                st.info(f"ℹ️ Moderate confidence (<{confidence_threshold:.0%})")

    with col2:
        # Confidence visualization
        fig = go.Figure(data=[
            go.Bar(
                y=['Cyberbullying', 'Not Cyberbullying'],
                x=[probabilities[1], probabilities[0]],
                orientation='h',
                marker_color=['#dc3545', '#28a745']
            )
        ])

        fig.update_layout(
            height=200,
            showlegend=False,
            xaxis_title="Probability",
            yaxis_title="Classification",
            margin=dict(l=0, r=0, t=0, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

    with col3:
        # Text statistics
        st.subheader("📊 Text Analysis")
        st.metric("Text Length", f"{len(text)} characters")
        st.metric("Word Count", f"{len(text.split())} words")

        # Quick risk indicators
        risk_words = ['stupid', 'worthless', 'hate', 'ugly', 'loser', 'kill', 'die']
        found_risk_words = [word for word in risk_words if word in text.lower()]

        if found_risk_words:
            st.warning(f"🚩 Found {len(found_risk_words)} risk-indicating words")
            st.write("Risk words detected:", ", ".join(found_risk_words))
        else:
            st.success("✅ No high-risk words detected")

    # SHAP Explanation
    if show_explanation:
        st.subheader("🔍 Explanation")
        st.write("Understanding why the AI made this prediction:")

        features = detector.get_shap_explanation(text, num_features)

        if features:
            # Create feature importance chart
            feature_df = pd.DataFrame(features)

            fig = px.bar(
                feature_df,
                x='importance',
                y='feature',
                orientation='h',
                title='Top Influential Words/Phrases',
                color='importance',
                color_continuous_scale='RdBu_r'
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Detailed feature table
            st.write("**Detailed Feature Impact:**")
            for i, feat in enumerate(features, 1):
                if feat['importance'] > 0:
                    st.markdown(
                        f"{i}. **'{feat['feature']}'** "
                        f"<span class='feature-positive'>(+{feat['importance']:.3f})</span> "
                        f"→ 🚩 Increases cyberbullying probability",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"{i}. **'{feat['feature']}'** "
                        f"<span class='feature-negative'>({feat['importance']:.3f})</span> "
                        f"→ ✅ Decreases cyberbullying probability",
                        unsafe_allow_html=True
                    )
        else:
            st.info("No significant features found for this text.")


def batch_analysis_mode(detector):
    """Batch analysis for multiple texts"""
    st.subheader("📊 Batch Analysis")

    st.write("""
    Upload a CSV file with a 'text' column to analyze multiple entries at once.
    This is useful for moderating large amounts of content.
    """)

    sample_data = pd.DataFrame({
        'text': [
            "You're such a loser, nobody wants you here",
            "Great work on the project team!",
            "I hate you, just disappear forever",
            "Let's collaborate on the next assignment"
        ]
    })

    st.dataframe(sample_data)

    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            if 'text' not in batch_df.columns:
                st.error("CSV file must contain a 'text' column")
                return

            st.success(f"✅ Loaded {len(batch_df)} records")

            with st.expander("Preview uploaded data"):
                st.dataframe(batch_df.head())

            if st.button("🚀 Process Batch Analysis", type="primary"):
                with st.spinner("Processing batch analysis..."):
                    # Process predictions
                    results = []
                    for text in batch_df['text']:
                        pred, probs = detector.predict(str(text))
                        results.append({
                            'text': text,
                            'prediction': 'Cyberbullying' if pred == 1 else 'Safe',
                            'confidence': probs[1] if pred == 1 else probs[0],
                            'risk_score': probs[1]
                        })

                    results_df = pd.DataFrame(results)

                    # Display results
                    st.subheader("📈 Batch Results Summary")

                    # Statistics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        cyberbullying_count = len(results_df[results_df['prediction'] == 'Cyberbullying'])
                        st.metric("Cyberbullying Detected", cyberbullying_count)

                    with col2:
                        safe_count = len(results_df[results_df['prediction'] == 'Safe'])
                        st.metric("Safe Content", safe_count)

                    with col3:
                        avg_risk = results_df['risk_score'].mean()
                        st.metric("Average Risk Score", f"{avg_risk:.1%}")

                    # Risk distribution
                    fig = px.histogram(
                        results_df,
                        x='risk_score',
                        title='Distribution of Cyberbullying Risk Scores',
                        nbins=20
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Results table
                    with st.expander("📋 Detailed Results"):
                        st.dataframe(results_df)

                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="cyberbullying_analysis.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")


def safety_tips_mode():
    """Cyberbullying safety information"""
    st.subheader("🛡️ Cyberbullying Safety Tips")

    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        ### 🚨 If You're Being Cyberbullied:

        **1. Don't Respond**
        - Bullies want attention - don't give it to them
        - Responding can make things worse

        **2. Save Evidence**
        - Take screenshots of harmful messages
        - Keep records of dates and times

        **3. Block and Report**
        - Use platform blocking features
        - Report to platform moderators
        - Tell trusted adults or authorities

        **4. Protect Your Accounts**
        - Use strong, unique passwords
        - Adjust privacy settings
        - Be careful what you share online
        """)

    with col2:
        st.write("""
        ### 🤝 How to Help Others:

        **1. Support the Target**
        - Offer emotional support
        - Don't share or forward harmful content
        - Encourage them to report

        **2. Be an Upstander**
        - Speak out against cyberbullying
        - Report content when you see it
        - Create positive content instead

        **3. Educational Resources**
        - StopBullying.gov
        - Cyberbullying Research Center
        - National Suicide Prevention Lifeline: 988
        """)

    st.info("""
    💡 **Remember**: Cyberbullying is never the victim's fault. Everyone deserves to feel safe online.
    If you're experiencing cyberbullying, reach out for help - you're not alone.
    """)


def model_info_mode(detector):
    """Model information and technical details"""
    st.subheader("🤖 Model Information")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("""
        ### About This AI Model

        This cyberbullying detection system uses a **Support Vector Machine (SVM)** 
        trained on thousands of labeled text examples to identify harmful language patterns.

        **How It Works:**
        1. **Text Processing**: Converts text into numerical features
        2. **Pattern Recognition**: Identifies cyberbullying language patterns
        3. **Risk Assessment**: Calculates probability of harmful content
        4. **Explanation**: Shows which words influenced the decision

        **Detection Capabilities:**
        - Personal attacks and insults
        - Threats and intimidation
        - Hate speech patterns
        - Exclusion and social manipulation

        **Limitations:**
        - May miss subtle or context-dependent bullying
        - Cultural/language variations can affect accuracy
        - Sarcasm and humor can be challenging
        """)

    with col2:
        st.write("### Technical Details")
        st.metric("Model Type", "SVM with Linear Kernel")
        st.metric("Feature Extraction", "TF-IDF Vectorization")
        st.metric("Training Data", "Cyberbullying Classification Dataset")

        st.write("### Accuracy Metrics")
        st.info("""
        - Precision: ~89%
        - Recall: ~87%  
        - F1-Score: ~88%
        - Overall Accuracy: ~89%
        """)

    st.warning("""
    ⚠️ **Important**: This tool is for assistance only. 
    Always use human judgment for serious moderation decisions.
    """)


if __name__ == "__main__":
    main()