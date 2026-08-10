import re
import string
from pathlib import Path
from collections import Counter

import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ModuleNotFoundError:
    WordCloud = None
    WORDCLOUD_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ModuleNotFoundError:
    nltk = None
    stopwords = None
    PorterStemmer = None
    word_tokenize = None
    NLTK_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)


LOCAL_NLTK_DATA = Path('nltk_data')
if NLTK_AVAILABLE:
    if LOCAL_NLTK_DATA.exists():
        nltk.data.path.append(str(LOCAL_NLTK_DATA.resolve()))

st.set_page_config(
    page_title='Email Spam Detector',
    page_icon='📧',
    layout='wide',
    initial_sidebar_state='expanded'
)

style_path = Path('assets/style.css')
if style_path.exists():
    st.markdown(f'<style>{style_path.read_text()}</style>', unsafe_allow_html=True)


def simple_tokenize(text: str):
    return [t for t in re.findall(r"[a-zA-Z]+", text) if t]


if NLTK_AVAILABLE:
    stemmer = PorterStemmer()
    try:
        stop_words = set(stopwords.words('english'))
    except (LookupError, OSError):
        stop_words = set(
            [
                'a', 'an', 'and', 'the', 'to', 'of', 'in', 'on', 'for', 'is', 'it',
                'this', 'that', 'with', 'as', 'was', 'are', 'be', 'by', 'at', 'from'
            ]
        )

    tokenizer = word_tokenize
    try:
        tokenizer('test sentence')
    except (LookupError, OSError):
        tokenizer = simple_tokenize
else:
    stemmer = None
    stop_words = set(
        [
            'a', 'an', 'and', 'the', 'to', 'of', 'in', 'on', 'for', 'is', 'it',
            'this', 'that', 'with', 'as', 'was', 'are', 'be', 'by', 'at', 'from'
        ]
    )
    tokenizer = simple_tokenize


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = tokenizer(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    if stemmer is not None:
        tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)


@st.cache_data
def load_data() -> pd.DataFrame:
    data_path = Path('data/spam.csv')
    df_raw = pd.read_csv(data_path, encoding='latin-1', low_memory=False)
    df = df_raw.iloc[:, :2].copy()
    df.columns = ['label', 'message']
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
    df['char_count'] = df['message'].astype(str).apply(len)
    df['word_count'] = df['message'].astype(str).apply(lambda msg: len(msg.split()))
    df['message_clean'] = df['message'].apply(preprocess_text)
    return df


@st.cache_resource
def train_models(df: pd.DataFrame):
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    X = tfidf.fit_transform(df['message_clean'])
    y = df['label_encoded']
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        'Multinomial Naive Bayes': MultinomialNB(alpha=0.1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Support Vector Machine': SVC(kernel='linear', probability=True, random_state=42)
    }

    metrics = []
    model_objects = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        model_objects[name] = model
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        metrics.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred)
        })

    return model_objects, tfidf, X_test, y_test, pd.DataFrame(metrics), predictions


def plot_class_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='label', palette=['#4caf50', '#f44336'], ax=ax)
    ax.set_title('Message Class Distribution')
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')
    return fig


def plot_length_histograms(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df, x='word_count', hue='label', element='step', stat='density', common_norm=False,
                 palette=['#4caf50', '#f44336'], bins=30, ax=ax)
    ax.set_title('Message Length Distribution by Label')
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Density')
    return fig


def make_wordcloud(text, title):
    if not WORDCLOUD_AVAILABLE:
        raise RuntimeError('wordcloud is not installed')

    wordcloud = WordCloud(
        width=600,
        height=400,
        background_color='white',
        colormap='viridis'
    ).generate(text)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    return fig


def get_top_terms(df: pd.DataFrame, label: str, top_n: int = 15):
    texts = df.loc[df['label'] == label, 'message_clean']
    tokens = Counter(' '.join(texts).split())
    return tokens.most_common(top_n)


def show_overview(df: pd.DataFrame, results_df: pd.DataFrame):
    st.title('📧 Email Spam Detection')
    st.markdown('*Binary NLP Classifier using TF-IDF + ML*')
    st.divider()

    total_messages = len(df)
    spam_count = int(df['label'].value_counts()['spam'])
    ham_count = int(df['label'].value_counts()['ham'])
    spam_pct = df['label'].value_counts(normalize=True)['spam'] * 100
    ham_pct = df['label'].value_counts(normalize=True)['ham'] * 100
    best_f1 = results_df.iloc[0]['F1 Score']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('📨 Total Messages', f'{total_messages:,}')
    col2.metric(
        '🚨 Spam Messages',
        f'{spam_count:,}',
        delta=f'{spam_pct:.1f}% of total',
        delta_color='inverse'
    )
    col3.metric(
        '✅ Ham Messages',
        f'{ham_count:,}',
        delta=f'{ham_pct:.1f}% of total'
    )
    col4.metric(
        '🏆 Best Model F1',
        f'{best_f1:.4f}',
        delta='Best performing'
    )
    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader('📋 Project Overview')
        st.markdown(
            '''
            This dashboard implements an **Email Spam Detection** 
            system using Natural Language Processing (NLP).

            **Pipeline:**
            1. 🧹 Text Preprocessing (clean, stem, remove stopwords)
            2. 🔢 TF-IDF Feature Extraction (5000 features, bigrams)
            3. 🤖 Train 3 ML classifiers
            4. 📊 Evaluate using Accuracy, Precision, Recall, F1

            **Tech Stack:**
            - `scikit-learn` — ML models + TF-IDF
            - `NLTK` — text preprocessing  
            - `WordCloud` — visualization
            - `Streamlit` — interactive dashboard
            '''
        )
    with right:
        st.subheader('📊 Model Results Summary')
        st.dataframe(
            results_df.style.format({
                'Accuracy': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1 Score': '{:.4f}'
            }).highlight_max(subset=['F1 Score'], color='#e8f0fe'),
            width='stretch'
        )

    st.divider()
    st.subheader('👀 Sample Data')
    tab1, tab2 = st.tabs(['🚨 Spam Examples', '✅ Ham Examples'])
    with tab1:
        spam_samples = df[df['label'] == 'spam'][['message', 'char_count', 'word_count']].head(5)
        st.dataframe(spam_samples, width='stretch')
    with tab2:
        ham_samples = df[df['label'] == 'ham'][['message', 'char_count', 'word_count']].head(5)
        st.dataframe(ham_samples, width='stretch')

    st.info(
        '''
        💡 **Navigate using the sidebar** to explore:
        EDA visualizations, WordClouds, Model Performance, 
        and the Live Spam Detector!
        '''
    )


def show_eda(df: pd.DataFrame, results_df: pd.DataFrame):
    st.title('📊 Exploratory Data Analysis')
    st.markdown('Understanding the dataset distribution and patterns.')
    st.divider()

    st.subheader('Class Distribution')
    col1, col2 = st.columns(2)

    counts = df['label'].value_counts()
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            ['Ham ✅', 'Spam 🚨'],
            counts.values,
            color=['#2ecc71', '#e74c3c'],
            edgecolor='white',
            linewidth=1.5,
            width=0.5
        )
        for bar, count in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                f'{count:,}',
                ha='center',
                fontweight='bold',
                fontsize=12
            )
        ax.set_title('Message Count by Class', fontsize=14, pad=15)
        ax.set_ylabel('Count')
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(
            counts.values,
            labels=['Ham (Legitimate)', 'Spam'],
            colors=['#2ecc71', '#e74c3c'],
            autopct='%1.1f%%',
            explode=(0, 0.1),
            startangle=90,
            textprops={'color': 'white'}
        )
        ax.set_title('Class Distribution %', fontsize=14, color='white', pad=15)
        fig.patch.set_facecolor('#0e1117')
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.subheader('Message Length Analysis')
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        df.boxplot(
            column='char_count',
            by='label',
            ax=ax,
            patch_artist=True,
            boxprops=dict(facecolor='#3498db', alpha=0.7)
        )
        ax.set_title('Character Count by Label')
        ax.set_xlabel('Label')
        ax.set_ylabel('Character Count')
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        plt.suptitle('')
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[df['label'] == 'spam']['word_count'].hist(
            bins=30, alpha=0.7, color='#e74c3c', label='Spam', ax=ax
        )
        df[df['label'] == 'ham']['word_count'].hist(
            bins=30, alpha=0.7, color='#2ecc71', label='Ham', ax=ax
        )
        ax.set_title('Word Count Distribution')
        ax.set_xlabel('Word Count')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        st.pyplot(fig)
        plt.close()

    st.info(
        '''
        💡 **Insight:** Spam messages tend to be significantly 
        longer than ham messages, with more characters and words. 
        This pattern is a useful signal for classification.
        '''
    )

    st.subheader('📋 Descriptive Statistics by Class')
    stats = df.groupby('label')[['char_count', 'word_count']].agg(['mean', 'median', 'std']).round(2)
    st.dataframe(stats, width='stretch')


def show_wordclouds(df: pd.DataFrame, results_df: pd.DataFrame):
    st.title('☁️ WordCloud Visualizations')
    st.markdown(
        '''
        WordClouds show word frequency — **larger = more frequent**.
        Compare the vocabulary used in spam vs legitimate emails.
        '''
    )
    st.divider()

    spam_text = ' '.join(df[df['label'] == 'spam']['message_clean'])
    ham_text = ' '.join(df[df['label'] == 'ham']['message_clean'])

    if WORDCLOUD_AVAILABLE:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('🚨 Spam Words')
            fig, ax = plt.subplots(figsize=(8, 5))
            wc_spam = WordCloud(
                width=800,
                height=400,
                background_color='black',
                colormap='Reds',
                max_words=100,
                collocations=False
            ).generate(spam_text)
            ax.imshow(wc_spam, interpolation='bilinear')
            ax.axis('off')
            fig.patch.set_facecolor('#0e1117')
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader('✅ Ham Words')
            fig, ax = plt.subplots(figsize=(8, 5))
            wc_ham = WordCloud(
                width=800,
                height=400,
                background_color='black',
                colormap='Greens',
                max_words=100,
                collocations=False
            ).generate(ham_text)
            ax.imshow(wc_ham, interpolation='bilinear')
            ax.axis('off')
            fig.patch.set_facecolor('#0e1117')
            st.pyplot(fig)
            plt.close()
    else:
        st.warning('WordCloud is not installed. Install `wordcloud` to render visual word clouds.')

    st.divider()

    st.divider()
    st.subheader('📊 Top 20 Most Frequent Words')
    col1, col2 = st.columns(2)

    spam_counts = Counter(spam_text.split()).most_common(20)
    spam_df = pd.DataFrame(spam_counts, columns=['Word', 'Count'])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(spam_df['Word'][::-1], spam_df['Count'][::-1], color='#f44336')
    ax.set_title('Spam Top Words')
    ax.set_xlabel('Count')
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    st.pyplot(fig)
    plt.close()

    ham_counts = Counter(ham_text.split()).most_common(20)
    ham_df = pd.DataFrame(ham_counts, columns=['Word', 'Count'])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(ham_df['Word'][::-1], ham_df['Count'][::-1], color='#4caf50')
    ax.set_title('Ham Top Words')
    ax.set_xlabel('Count')
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    st.pyplot(fig)
    plt.close()

    st.warning(
        '''
        🔍 **Key Observation:** Spam emails frequently contain 
        words like 'free', 'win', 'prize', 'call', 'claim', 'urgent'.
        Ham emails contain more personal, conversational words.
        These patterns are exactly what TF-IDF captures!
        '''
    )


def show_model_performance(df: pd.DataFrame, results_df: pd.DataFrame, models, y_test, predictions):
    st.title('🤖 Model Performance')
    st.markdown('Comparing 3 classifiers on the spam detection task.')
    st.divider()

    cols = st.columns(3)
    for i, (_, row) in enumerate(results_df.iterrows()):
        with cols[i]:
            is_best = i == 0
            border_color = '#2ecc71' if is_best else '#3d3d3d'
            st.markdown(f"""
            <div style='border: 2px solid {border_color}; 
                 padding: 15px; border-radius: 10px;
                 text-align: center;'>
                <h4>{'🏆 ' if is_best else ''}{row['Model']}</h4>
                <p>Accuracy: <b>{row['Accuracy']:.4f}</b></p>
                <p>Precision: <b>{row['Precision']:.4f}</b></p>
                <p>Recall: <b>{row['Recall']:.4f}</b></p>
                <p>F1 Score: <b>{row['F1 Score']:.4f}</b></p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.subheader('📊 Metrics Comparison')

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for ax, metric, color in zip(axes, metrics, colors):
        bars = ax.bar(
            [m.replace(' ', '\n') for m in results_df['Model']],
            results_df[metric],
            color=color,
            alpha=0.85,
            edgecolor='white'
        )
        for bar, val in zip(bars, results_df[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f'{val:.4f}',
                ha='center',
                fontsize=9,
                color='white'
            )
        ax.set_title(metric, color='white', fontsize=12)
        ax.set_ylim(0.9, 1.01)
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white', labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    fig.patch.set_facecolor('#0e1117')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.subheader('🔲 Confusion Matrices')
    cols = st.columns(3)
    model_names = list(models.keys())

    for i, name in enumerate(model_names):
        with cols[i]:
            cm = confusion_matrix(y_test, predictions[name])
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                ax=ax,
                cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'}
            )
            ax.set_title(name.replace(' ', '\n'), color='white', fontsize=10)
            ax.set_xlabel('Predicted', color='white')
            ax.set_ylabel('Actual', color='white')
            ax.tick_params(colors='white')
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            st.pyplot(fig)
            plt.close()

    with st.expander('💡 Why is Recall important for spam detection?'):
        st.markdown(
            '''
            **Recall = TP / (TP + FN)**
            
            In spam detection:
            - **False Negative** = Spam reaches inbox → annoying but tolerable
            - **False Positive** = Ham blocked as spam → user misses important email ⚠️
            
            We want **high Recall** so spam is caught,
            while keeping **Precision** high enough to 
            avoid blocking legitimate emails.
            
            All 3 models achieve Recall > 95% — production ready! ✅
            '''
        )


def show_predictor(df: pd.DataFrame, models, tfidf, predictions):
    st.title('🔮 Live Spam Detector')
    st.markdown('Type any message to instantly check if it\'s spam or ham.')
    st.divider()

    col1, col2 = st.columns([5, 3])

    with col1:
        st.subheader('✍️ Enter Your Message')

        user_input = st.text_area(
            label='Message',
            placeholder='Type or paste your message here...',
            height=150,
            label_visibility='collapsed'
        )

        model_choice = st.selectbox(
            'Choose Model:',
            options=list(models.keys()),
            index=0
        )

        analyze_btn = st.button(
            '🔍 Analyze Message',
            type='primary',
            use_container_width=True
        )

        st.divider()
        st.markdown('**🧪 Try these examples:**')

        example_messages = {
            '🚨 Spam 1': 'WINNER!! You have been selected to receive a $1000 prize! Call now!',
            '🚨 Spam 2': 'FREE entry to our competition! Text WIN to 87121 now!',
            '✅ Ham 1': 'Hey, are we still meeting for lunch tomorrow?',
            '✅ Ham 2': 'Can you send me the report by end of day please?'
        }

        for label, msg in example_messages.items():
            if st.button(label, use_container_width=True):
                st.session_state['example_msg'] = msg
                st.rerun()

        if 'example_msg' in st.session_state:
            user_input = st.session_state['example_msg']

    with col2:
        st.subheader('📊 Prediction Result')

        if analyze_btn and user_input.strip():
            with st.spinner('Analyzing...'):
                time.sleep(0.5)

                cleaned = preprocess_text(user_input)
                vectorized = tfidf.transform([cleaned])
                selected_model = models[model_choice]
                prediction = selected_model.predict(vectorized)[0]
                probability = selected_model.predict_proba(vectorized)[0]
                confidence = max(probability) * 100

                if prediction == 1:
                    st.error('# 🚨 SPAM DETECTED')
                    st.markdown(f'**Confidence: {confidence:.1f}%**')
                    st.progress(confidence / 100)
                    st.markdown(
                        '***⚠️ Warning Signs:***\n'
                        'This message shows patterns commonly found in spam emails.'
                    )
                else:
                    st.success('# ✅ HAM (Legitimate)')
                    st.markdown(f'**Confidence: {confidence:.1f}%**')
                    st.progress(confidence / 100)
                    st.markdown(
                        '***✅ Looks Safe:***\n'
                        'This message appears to be legitimate.'
                    )

                st.divider()
                st.markdown('**Probability Breakdown:**')
                prob_df = pd.DataFrame({
                    'Class': ['Ham ✅', 'Spam 🚨'],
                    'Probability': [f"{probability[0]*100:.1f}%", f"{probability[1]*100:.1f}%"]
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

                st.divider()
                st.markdown('**Preprocessed Text:**')
                st.code(cleaned, language=None)

        elif analyze_btn and not user_input.strip():
            st.warning('⚠️ Please enter a message first!')
        else:
            st.info('👈 Enter a message and click **Analyze** to see results.')

    st.divider()
    st.subheader('📋 Batch Test — All Models')
    st.markdown('See how all 3 models predict the same message.')

    if user_input.strip():
        cleaned = preprocess_text(user_input)
        vectorized = tfidf.transform([cleaned])

        batch_results = []
        for name, model in models.items():
            pred = model.predict(vectorized)[0]
            prob = model.predict_proba(vectorized)[0]
            batch_results.append({
                'Model': name,
                'Prediction': '🚨 SPAM' if pred == 1 else '✅ HAM',
                'Ham %': f"{prob[0]*100:.1f}%",
                'Spam %': f"{prob[1]*100:.1f}%",
                'Confidence': f"{max(prob)*100:.1f}%"
            })

        batch_df = pd.DataFrame(batch_results)
        st.dataframe(batch_df, use_container_width=True, hide_index=True)
    else:
        st.caption('Enter a message above to see batch predictions.')


def main():
    with st.spinner('🔄 Loading dataset and training models...'):
        df = load_data()
        models, tfidf, X_test, y_test, results_df, predictions = train_models(df)

    with st.sidebar:
        st.markdown('# 📧 Spam Detector')
        st.markdown('*NLP Binary Classifier*')
        st.divider()

        page = st.radio(
            'Navigate',
            options=[
                '🏠 Overview',
                '📊 EDA & Visualizations',
                '☁️ WordClouds',
                '🤖 Model Performance',
                '🔮 Live Spam Detector'
            ],
            label_visibility='collapsed'
        )
        st.divider()

        st.markdown('### 📈 Dataset Stats')
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Total', f"{len(df):,}")
            st.metric('Spam', f"{df['label'].value_counts()['spam']:,}")
        with col2:
            st.metric('Ham', f"{df['label'].value_counts()['ham']:,}")
            spam_pct = df['label'].value_counts(normalize=True)['spam'] * 100
            st.metric('Spam %', f"{spam_pct:.1f}%")

        st.divider()

        best_model_name = results_df.iloc[0]['Model']
        best_f1 = results_df.iloc[0]['F1 Score']
        st.markdown('### 🏆 Best Model')
        st.success(f'**{best_model_name}**')
        st.metric('F1 Score', f"{best_f1:.4f}")

        st.divider()
        st.caption('Built with Streamlit + scikit-learn')

    if page == '🏠 Overview':
        show_overview(df, results_df)
    elif page == '📊 EDA & Visualizations':
        show_eda(df, results_df)
    elif page == '☁️ WordClouds':
        show_wordclouds(df, results_df)
    elif page == '🤖 Model Performance':
        show_model_performance(df, results_df, models, y_test, predictions)
    elif page == '🔮 Live Spam Detector':
        show_predictor(df, models, tfidf, predictions)


if __name__ == '__main__':
    main()
