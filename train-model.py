import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Create model directory
os.makedirs('model', exist_ok=True)


# ---------------------
# LOAD DATA - FIXED
# ---------------------
def load_data():
    df = pd.read_csv('data/cyberbullying_tweets.csv')
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print("Original distribution:")
    print(df['cyberbullying_type'].value_counts())
    return df


# ---------------------
# PREPROCESS - ENHANCED
# ---------------------
def clean_text(text):
    """Enhanced text cleaning"""
    if not isinstance(text, str):
        return ""

    # Remove URLs, mentions
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)  # Keep hashtag words

    # Keep some punctuation that matters for sentiment
    text = re.sub(r'[^\w\s!?]', ' ', text)

    # Lowercase and remove extra spaces
    text = text.lower().strip()
    text = ' '.join(text.split())

    return text


def validate_dataset(df):
    """CRITICAL: Check for data quality issues"""
    print("\n" + "=" * 70)
    print("🔍 DATASET VALIDATION")
    print("=" * 70)

    # Check samples from each class
    for cyber_type in df['cyberbullying_type'].unique():
        samples = df[df['cyberbullying_type'] == cyber_type].head(5)
        print(f"\n📋 Sample '{cyber_type}' tweets:")
        for _, row in samples.iterrows():
            print(f"  - '{row['tweet_text'][:80]}...'")

    # Check for contamination in not_cyberbullying
    not_cyber = df[df['cyberbullying_type'] == 'not_cyberbullying']['tweet_text']

    # Keywords that shouldn't be in non-bullying
    bullying_keywords = ['stupid', 'idiot', 'hate', 'ugly', 'worthless', 'kill',
                         'die', 'loser', 'trash', 'shit', 'bitch', 'whore']

    contaminated_count = 0
    print("\n⚠️  Checking for contamination in 'not_cyberbullying' class:")
    for keyword in bullying_keywords:
        count = not_cyber.str.contains(keyword, case=False, na=False).sum()
        if count > 0:
            contaminated_count += count
            print(f"  - '{keyword}': {count} instances")

    if contaminated_count > 0:
        print(f"\n❌ WARNING: Found {contaminated_count} potentially mislabeled samples!")
        print("   Consider manual review of 'not_cyberbullying' class")
    else:
        print("\n✅ No obvious contamination detected")


def preprocess_data(df):
    df = df.dropna()
    df['tweet_text'] = df['tweet_text'].astype(str)

    # Clean the text
    df['tweet_text'] = df['tweet_text'].apply(clean_text)

    # Remove empty texts after cleaning
    df = df[df['tweet_text'].str.len() > 0]

    # Validate dataset quality
    validate_dataset(df)

    # Binary label - VERIFIED
    df['is_cyberbullying'] = df['cyberbullying_type'].apply(
        lambda x: 0 if x == 'not_cyberbullying' else 1
    )

    print("\n" + "=" * 70)
    print("📊 CLASS DISTRIBUTION")
    print("=" * 70)
    print(df['is_cyberbullying'].value_counts())
    print(
        f"Imbalance ratio: {df['is_cyberbullying'].value_counts()[1] / df['is_cyberbullying'].value_counts()[0]:.2f}:1")

    return df


# ---------------------
# TRAIN MODEL - COMPLETELY FIXED
# ---------------------
def train_model():
    df = load_data()
    df = preprocess_data(df)

    X = df['tweet_text']
    y = df['is_cyberbullying']

    # FIXED: Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n" + "=" * 70)
    print("🚀 BUILDING IMPROVED MODEL WITH SMOTE")
    print("=" * 70)

    # FIXED: Use class_weight and better parameters
    base_svm = LinearSVC(
        class_weight='balanced',  # Handle imbalance
        C=1.0,
        max_iter=3000,
        random_state=42,
        dual=False  # Better for n_samples > n_features
    )

    calibrated_svm = CalibratedClassifierCV(base_svm, cv=3, method='sigmoid')

    # FIXED: Improved TF-IDF with better parameters
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english'
    )

    # FIXED: Use SMOTE to handle imbalance (upsample minority class)
    # This creates synthetic samples instead of throwing away data
    smote = SMOTE(random_state=42, k_neighbors=5)

    # Create imbalanced-learn pipeline
    pipeline = ImbPipeline([
        ('tfidf', tfidf),
        ('smote', smote),  # Apply SMOTE after vectorization
        ('clf', calibrated_svm)
    ])

    print("\n⏳ Training model with SMOTE...")
    print(f"Training samples: {len(X_train)}")
    print(f"  - Cyberbullying: {y_train.sum()}")
    print(f"  - Not Cyberbullying: {(y_train == 0).sum()}")

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluation
    print("\n" + "=" * 70)
    print("📊 MODEL PERFORMANCE")
    print("=" * 70)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Not Cyberbullying', 'Cyberbullying'],
        digits=3
    ))

    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\n🎯 ROC-AUC Score: {roc_auc:.3f}")

    # Test obvious cases
    print("\n" + "=" * 70)
    print("🧪 TESTING OBVIOUS CASES")
    print("=" * 70)

    test_cases = [
        ("you are stupid and ugly", 1),
        ("have a nice day", 0),
        ("i hate you loser", 1),
        ("good morning everyone", 0),
        ("kill yourself worthless trash", 1),
        ("hope you have a wonderful day", 0)
    ]

    correct = 0
    for text, expected in test_cases:
        prob = pipeline.predict_proba([text])[0][1]
        pred = pipeline.predict([text])[0]
        result = "✅" if pred == expected else "❌"
        print(f"{result} '{text}' -> {pred} ({prob:.3f}) [Expected: {expected}]")
        if pred == expected:
            correct += 1

    print(f"\n🎯 Obvious Cases Accuracy: {correct}/{len(test_cases)} ({correct / len(test_cases) * 100:.1f}%)")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Cyberbullying', 'Cyberbullying'],
                yticklabels=['Not Cyberbullying', 'Cyberbullying'])
    plt.title(f"Confusion Matrix (ROC-AUC: {roc_auc:.3f})")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Add percentages
    for i in range(2):
        for j in range(2):
            plt.text(j + 0.5, i + 0.7, f'({cm[i, j] / cm[i].sum() * 100:.1f}%)',
                     ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig('model/confusion_matrix.png', dpi=150)
    plt.show()

    return pipeline, X_train, y_train


# ---------------------
# EXPLANATION - FIXED
# ---------------------
def explain_single_prediction(pipeline, sample_text):
    """Enhanced explanation"""
    sample_text_clean = clean_text(sample_text)

    # Get pipeline components (handle ImbPipeline)
    vectorizer = pipeline.named_steps['tfidf']
    calibrated_model = pipeline.named_steps['clf']
    base_estimator = calibrated_model.calibrated_classifiers_[0].estimator

    # Transform text
    sample_vec = vectorizer.transform([sample_text_clean])

    # Get prediction
    prediction = pipeline.predict([sample_text_clean])[0]
    probability = pipeline.predict_proba([sample_text_clean])[0][1]

    # Get feature names and coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = base_estimator.coef_[0]
    intercept = base_estimator.intercept_[0]

    # Calculate contributions
    feature_contributions = []
    total_score = intercept

    nonzero_indices = sample_vec.nonzero()[1]

    for idx in nonzero_indices:
        feature_name = feature_names[idx]
        feature_value = sample_vec[0, idx]
        feature_coef = coefficients[idx]
        contribution = feature_value * feature_coef
        total_score += contribution

        feature_contributions.append({
            'feature': feature_name,
            'value': feature_value,
            'coefficient': feature_coef,
            'contribution': contribution
        })

    feature_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)

    return {
        'text': sample_text,
        'cleaned_text': sample_text_clean,
        'prediction': prediction,
        'probability': probability,
        'decision_score': total_score,
        'feature_contributions': feature_contributions[:15],
        'prediction_label': 'Cyberbullying' if prediction == 1 else 'Not Cyberbullying',
        'intercept': intercept
    }


# ---------------------
# VISUALIZE EXPLANATION
# ---------------------
def visualize_explanation(explanation):
    contributions = explanation['feature_contributions']

    if not contributions:
        print("No features to visualize.")
        return

    features = [x['feature'] for x in contributions]
    values = [x['contribution'] for x in contributions]
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in values]

    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(features))

    bars = plt.barh(y_pos, values, color=colors, alpha=0.7)
    plt.yticks(y_pos, features, fontsize=10)
    plt.xlabel('Contribution to Prediction', fontsize=12)

    title = (f"Explanation: {explanation['text'][:50]}...\n"
             f"Prediction: {explanation['prediction_label']} "
             f"(Confidence: {explanation['probability']:.1%})")

    plt.title(title, fontsize=14, weight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.7, label='Increases Cyberbullying Score'),
        Patch(facecolor='#3498db', alpha=0.7, label='Decreases Cyberbullying Score')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.show()


# ---------------------
# PRINT TEXT ANALYSIS
# ---------------------
def print_text_analysis(explanation):
    print(f"\n{'=' * 70}")
    print(f"🔍 Analyzing: '{explanation['text']}'")
    print(f"{'=' * 70}")
    print(f"🧹 Cleaned: '{explanation['cleaned_text']}'")
    print(f"🎯 Prediction: {explanation['prediction_label']}")
    print(f"📊 Confidence: {explanation['probability']:.1%}")
    print(f"📈 Decision Score: {explanation['decision_score']:.3f}")

    print(f"\n📋 Top Contributing Features:")
    print("-" * 70)

    contributions = explanation['feature_contributions']
    for i, contrib in enumerate(contributions[:10], 1):
        feature = contrib['feature']
        impact = contrib['contribution']
        arrow = "🔴" if impact > 0 else "🔵"
        direction = "BULLYING" if impact > 0 else "NOT BULLYING"

        print(f"{i:2d}. {arrow} {feature:25s} → {direction:12s} (Impact: {impact:+.4f})")

    print("-" * 70)


# ---------------------
# SAVE MODEL
# ---------------------
def save_model(pipeline):
    with open('model/cyberbully_model_fixed.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("\n💾 Model saved to 'model/cyberbully_model_fixed.pkl'")


# ---------------------
# MAIN - FIXED
# ---------------------
if __name__ == '__main__':
    print("=" * 70)
    print("🚀 CYBERBULLYING DETECTION MODEL - FIXED VERSION")
    print("=" * 70)

    # Train and save model
    pipeline, X_train, y_train = train_model()
    save_model(pipeline)

    # Enhanced test cases
    test_cases = [
        # Clear bullying
        "You're so stupid and worthless, nobody likes you",
        "I hope you die you useless piece of trash",
        "You're such a loser and everyone hates you",
        "Nobody will ever love someone as ugly as you",
        "Kill yourself you worthless piece of shit",
        "You're a disgusting human being",

        # Clear non-bullying
        "Great job on the project! Really impressive work",
        "Looking forward to our meeting tomorrow",
        "That's an interesting point, thanks for sharing",
        "Have a wonderful day everyone!",
        "I appreciate your help with this matter",
    ]

    print("\n" + "=" * 70)
    print("🧪 DETAILED EXPLANATIONS")
    print("=" * 70)

    for i, test_text in enumerate(test_cases[:6], 1):  # First 6 cases
        explanation = explain_single_prediction(pipeline, test_text)
        print_text_analysis(explanation)

        # Visualize first 3
        if i <= 3:
            visualize_explanation(explanation)

    print("\n✅ Model training and testing complete!")
    print("📁 Files saved:")
    print("   - model/cyberbully_model_fixed.pkl")
    print("   - model/confusion_matrix.png")
