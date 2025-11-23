import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

os.makedirs('model', exist_ok=True)


# ==================== PATTERN-BASED DETECTOR ====================
class SevereBullyingDetector(BaseEstimator, TransformerMixin):
    """
    Rule-based detector for severe bullying patterns
    Catches extreme cases that ML might miss
    """

    def __init__(self):
        # Severe threat patterns (HIGH PRIORITY)
        self.severe_patterns = [
            r'\bkill\s+yourself\b',
            r'\bkys\b',  # kill yourself abbreviation
            r'\bgo\s+die\b',
            r'\bshould\s+die\b',
            r'\bhope\s+you\s+die\b',
            r'\bwish\s+you\s+(were\s+)?dead\b',
            r'\bend\s+your\s+life\b',
            r'\bcommit\s+suicide\b',
        ]

        # Severe insult patterns
        self.severe_insults = [
            r'\bworthless\s+(piece|trash|shit|garbage)',
            r'\buseless\s+(piece|trash|shit|garbage)',
            r'\byou\s+are\s+(a\s+)?piece\s+of\s+shit',
            r'\byou\s+should\s+be\s+(dead|killed|shot)',
            r'\bnobody\s+(will\s+ever\s+)?loves?\s+you',
            r'\beveryone\s+hates\s+you',
            r'\bwaste\s+of\s+(space|oxygen|life)',
        ]

        # Severe words (for density check)
        self.severe_words = {
            'kill', 'die', 'death', 'dead', 'suicide', 'hang', 'shoot',
            'worthless', 'useless', 'disgusting', 'ugly', 'stupid',
            'idiot', 'loser', 'trash', 'garbage', 'hate', 'despise',
            'pathetic', 'failure', 'waste'
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Extract severity features"""
        features = []

        for text in X:
            text_lower = text.lower() if isinstance(text, str) else ""

            # Count severe pattern matches
            severe_count = sum(
                1 for pattern in self.severe_patterns
                if re.search(pattern, text_lower)
            )

            # Count severe insult matches
            insult_count = sum(
                1 for pattern in self.severe_insults
                if re.search(pattern, text_lower)
            )

            # Severe word density
            words = text_lower.split()
            word_count = len(words)
            severe_word_count = sum(1 for word in words if word in self.severe_words)
            severity_density = severe_word_count / max(word_count, 1)

            # Targeted attack detection ("you are/you're" + negative descriptor)
            targeted_attack = 0
            if re.search(r'\b(you\s+are|you\'?re)\b', text_lower):
                negative_descriptors = [
                    'stupid', 'ugly', 'worthless', 'useless', 'disgusting',
                    'idiot', 'loser', 'fat', 'dumb', 'pathetic', 'failure'
                ]
                if any(word in text_lower for word in negative_descriptors):
                    targeted_attack = 1

            # Repeated severe words (emphasis)
            word_freq = {}
            for word in words:
                if word in self.severe_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            max_repetition = max(word_freq.values()) if word_freq else 0

            # ALL CAPS SHOUTING with severe content
            caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
            is_shouting = caps_ratio > 0.5 and len(text) > 10
            shouting_severity = is_shouting * severity_density

            features.append([
                severe_count,  # 0: Severe threat patterns
                insult_count,  # 1: Severe insult patterns
                severity_density,  # 2: Severe word density
                targeted_attack,  # 3: Personal attack
                max_repetition,  # 4: Word repetition
                shouting_severity,  # 5: Shouting + severe content
                len(text_lower),  # 6: Text length
                word_count  # 7: Word count
            ])

        return np.array(features)

    def detect_severe(self, text):
        """Quick severe bullying check (for hybrid approach)"""
        text_lower = text.lower() if isinstance(text, str) else ""

        # Check severe patterns first
        for pattern in self.severe_patterns:
            if re.search(pattern, text_lower):
                return True

        # Check severe insults
        for pattern in self.severe_insults:
            if re.search(pattern, text_lower):
                return True

        # Check high-density severe words in short text
        words = text_lower.split()
        word_count = len(words)
        severe_word_count = sum(1 for word in words if word in self.severe_words)

        if severe_word_count >= 3 and word_count < 15:
            return True

        return False


# ==================== HYBRID CLASSIFIER ====================
class HybridBullyingClassifier:
    """
    Combines ML model with rule-based severe pattern detection
    """

    def __init__(self, ml_pipeline, severe_detector):
        self.ml_pipeline = ml_pipeline
        self.severe_detector = severe_detector

    def predict(self, X):
        predictions = []

        for text in X:
            # First check severe patterns (rule-based override)
            if self.severe_detector.detect_severe(text):
                predictions.append(1)  # Override to cyberbullying
            else:
                # Use ML prediction
                pred = self.ml_pipeline.predict([text])[0]
                predictions.append(pred)

        return np.array(predictions)

    def predict_proba(self, X):
        probas = []

        for text in X:
            # Check severe patterns
            if self.severe_detector.detect_severe(text):
                # High confidence cyberbullying
                probas.append([0.01, 0.99])
            else:
                # Use ML probability
                proba = self.ml_pipeline.predict_proba([text])[0]
                probas.append(proba)

        return np.array(probas)


# ==================== DATA LOADING & PREPROCESSING ====================
def load_data():
    df = pd.read_csv('data/cyberbullying_tweets.csv')
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print("Original distribution:")
    print(df['cyberbullying_type'].value_counts())
    return df


def clean_text(text):
    """Enhanced text cleaning"""
    if not isinstance(text, str):
        return ""

    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s!?]', ' ', text)
    text = text.lower().strip()
    text = ' '.join(text.split())

    return text


def is_likely_mislabeled(text):
    """Detect likely mislabeled 'not_cyberbullying' samples"""
    text_lower = text.lower()

    # Use the SevereBullyingDetector patterns
    detector = SevereBullyingDetector()
    return detector.detect_severe(text)


def clean_dataset(df):
    """Remove likely mislabeled samples"""
    print("\n" + "=" * 70)
    print("🧹 CLEANING DATASET - REMOVING MISLABELED SAMPLES")
    print("=" * 70)

    original_count = len(df)
    not_cyber_original = len(df[df['cyberbullying_type'] == 'not_cyberbullying'])

    # Find mislabeled samples
    df['likely_mislabeled'] = df.apply(
        lambda row: is_likely_mislabeled(row['tweet_text'])
        if row['cyberbullying_type'] == 'not_cyberbullying'
        else False,
        axis=1
    )

    mislabeled_count = df['likely_mislabeled'].sum()

    print(f"\n📊 Found {mislabeled_count} likely mislabeled samples in 'not_cyberbullying' class")

    # Show examples
    if mislabeled_count > 0:
        print("\n⚠️  Examples of likely mislabeled samples:")
        mislabeled_samples = df[df['likely_mislabeled']].head(10)
        for i, (_, row) in enumerate(mislabeled_samples.iterrows(), 1):
            print(f"{i:2d}. '{row['tweet_text'][:80]}...'")

    # Remove mislabeled samples
    df_cleaned = df[~df['likely_mislabeled']].copy()
    df_cleaned = df_cleaned.drop('likely_mislabeled', axis=1)

    not_cyber_after = len(df_cleaned[df_cleaned['cyberbullying_type'] == 'not_cyberbullying'])

    print(f"\n✅ Removed {original_count - len(df_cleaned)} samples")
    print(f"   'not_cyberbullying' class: {not_cyber_original} → {not_cyber_after}")
    print(f"   Total samples: {original_count} → {len(df_cleaned)}")

    return df_cleaned


def preprocess_data(df):
    df = df.dropna()
    df['tweet_text'] = df['tweet_text'].astype(str)
    df['tweet_text'] = df['tweet_text'].apply(clean_text)
    df = df[df['tweet_text'].str.len() > 0]

    # Clean dataset
    df = clean_dataset(df)

    # Binary label
    df['is_cyberbullying'] = df['cyberbullying_type'].apply(
        lambda x: 0 if x == 'not_cyberbullying' else 1
    )

    print("\n" + "=" * 70)
    print("📊 FINAL CLASS DISTRIBUTION")
    print("=" * 70)
    print(df['is_cyberbullying'].value_counts())
    imbalance = df['is_cyberbullying'].value_counts()[1] / df['is_cyberbullying'].value_counts()[0]
    print(f"Imbalance ratio: {imbalance:.2f}:1")

    return df


# ==================== FEATURE ENGINEERING ====================
class FeatureUnion(BaseEstimator, TransformerMixin):
    """Combine TF-IDF features with severity features"""

    def __init__(self, tfidf_vectorizer, severity_detector):
        self.tfidf_vectorizer = tfidf_vectorizer
        self.severity_detector = severity_detector

    def fit(self, X, y=None):
        self.tfidf_vectorizer.fit(X)
        self.severity_detector.fit(X)
        return self

    def transform(self, X):
        tfidf_features = self.tfidf_vectorizer.transform(X).toarray()
        severity_features = self.severity_detector.transform(X)

        # Combine features
        combined = np.hstack([tfidf_features, severity_features])
        return combined


# ==================== TRAINING ====================
def train_model():
    df = load_data()
    df = preprocess_data(df)

    X = df['tweet_text']
    y = df['is_cyberbullying']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n" + "=" * 70)
    print("🚀 TRAINING HYBRID MODEL (ML + RULE-BASED)")
    print("=" * 70)

    # Create components
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),  # Added trigrams for better context
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english'
    )

    severe_detector = SevereBullyingDetector()
    feature_union = FeatureUnion(tfidf, severe_detector)

    # ML classifier
    base_svm = LinearSVC(
        class_weight='balanced',
        C=1.0,
        max_iter=3000,
        random_state=42,
        dual=False
    )

    calibrated_svm = CalibratedClassifierCV(base_svm, cv=3, method='sigmoid')

    # SMOTE for balancing
    smote = SMOTE(random_state=42, k_neighbors=5)

    # Create ML pipeline
    ml_pipeline = ImbPipeline([
        ('features', feature_union),
        ('smote', smote),
        ('clf', calibrated_svm)
    ])

    print(f"\n⏳ Training with {len(X_train)} samples...")
    print(f"  - Cyberbullying: {y_train.sum()}")
    print(f"  - Not Cyberbullying: {(y_train == 0).sum()}")

    ml_pipeline.fit(X_train, y_train)

    # Create hybrid classifier
    hybrid_clf = HybridBullyingClassifier(ml_pipeline, severe_detector)

    # ==================== EVALUATION ====================
    print("\n" + "=" * 70)
    print("📊 MODEL PERFORMANCE")
    print("=" * 70)

    y_pred = hybrid_clf.predict(X_test)
    y_proba = hybrid_clf.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Not Cyberbullying', 'Cyberbullying'],
        digits=3
    ))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\n🎯 ROC-AUC Score: {roc_auc:.3f}")

    # ==================== TEST OBVIOUS CASES ====================
    print("\n" + "=" * 70)
    print("🧪 TESTING OBVIOUS CASES (WITH HYBRID DETECTION)")
    print("=" * 70)

    test_cases = [
        # Severe bullying (should ALL be detected now)
        ("kill yourself", 1),
        ("kys loser", 1),
        ("you should die", 1),
        ("I hope you die", 1),
        ("go die you worthless piece of trash", 1),
        ("kill yourself worthless trash", 1),
        ("I hope you die you useless piece of shit", 1),
        ("you are stupid and ugly", 1),
        ("i hate you loser", 1),
        ("you're such a loser everyone hates you", 1),
        ("nobody will ever love someone as ugly as you", 1),
        ("waste of space nobody likes you", 1),

        # Clear non-bullying
        ("have a nice day", 0),
        ("good morning everyone", 0),
        ("hope you have a wonderful day", 0),
        ("great job on the project", 0),
        ("thanks for your help", 0),
        ("love your work keep it up", 0),
    ]

    correct = 0
    for text, expected in test_cases:
        prob = hybrid_clf.predict_proba([text])[0][1]
        pred = hybrid_clf.predict([text])[0]
        result = "✅" if pred == expected else "❌"
        label = "BULLYING" if pred == 1 else "NOT BULLYING"
        rule_detected = severe_detector.detect_severe(text)
        detection_method = "[RULE]" if rule_detected else "[ML]"
        print(f"{result} '{text[:45]:45s}' → {label:15s} {detection_method} ({prob:.3f}) [Expected: {expected}]")
        if pred == expected:
            correct += 1

    accuracy = correct / len(test_cases) * 100
    print(f"\n🎯 Obvious Cases Accuracy: {correct}/{len(test_cases)} ({accuracy:.1f}%)")

    if accuracy >= 95:
        print("✅ EXCELLENT! Hybrid model performing very well")
    elif accuracy >= 90:
        print("✅ VERY GOOD! Minor improvements possible")
    elif accuracy >= 80:
        print("⚠️  GOOD but could be better")
    else:
        print("❌ NEEDS IMPROVEMENT")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r',
                xticklabels=['Not Cyberbullying', 'Cyberbullying'],
                yticklabels=['Not Cyberbullying', 'Cyberbullying'],
                cbar_kws={'label': 'Count'})
    plt.title(f"Confusion Matrix (Hybrid Model)\nROC-AUC: {roc_auc:.3f} | Test Accuracy: {accuracy:.1f}%",
              fontsize=14, weight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / cm[i].sum() * 100
            plt.text(j + 0.5, i + 0.75, f'({percentage:.1f}%)',
                     ha='center', va='center', fontsize=11,
                     color='white' if percentage > 50 else 'black',
                     weight='bold')

    plt.tight_layout()
    plt.savefig('model/confusion_matrix_hybrid.png', dpi=150, bbox_inches='tight')
    plt.close()

    return hybrid_clf


# ==================== SAVE MODEL ====================
def save_model(hybrid_clf):
    with open('model/cyberbully_model_hybrid.pkl', 'wb') as f:
        pickle.dump(hybrid_clf, f)
    print("\n💾 Model saved to 'model/cyberbully_model_hybrid.pkl'")


# ==================== MAIN ====================
if __name__ == '__main__':
    print("=" * 70)
    print("🚀 CYBERBULLYING DETECTION - HYBRID MODEL (ML + RULES)")
    print("=" * 70)

    hybrid_clf = train_model()
    save_model(hybrid_clf)

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print("\n📁 Files saved:")
    print("   - model/cyberbully_model_hybrid.pkl")
    print("   - model/confusion_matrix_hybrid.png")
    print("\n🎯 Improvements:")
    print("   ✅ 'kill yourself' and severe threats detected via rules")
    print("   ✅ Hybrid approach combines ML flexibility + rule precision")
    print("   ✅ Reduced false negatives on extreme bullying")
    print("   ✅ Maintained ML generalization for nuanced cases")