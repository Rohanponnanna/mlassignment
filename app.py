import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Genre Classifier",
    page_icon="🎬",
    layout="centered"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0d0d0d;
        color: #f0ede6;
    }
    .main { background-color: #0d0d0d; }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #f5c842;
    }

    .stTextArea textarea {
        background-color: #1a1a1a !important;
        color: #f0ede6 !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 15px !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #f5c842, #e08c00);
        color: #0d0d0d;
        font-weight: 700;
        font-size: 16px;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .result-box {
        background: #1a1a1a;
        border-left: 4px solid #f5c842;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .genre-label {
        font-size: 2rem;
        font-family: 'Playfair Display', serif;
        color: #f5c842;
    }
    .confidence-bar {
        background: #2a2a2a;
        border-radius: 4px;
        height: 10px;
        margin: 6px 0 14px 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #f5c842, #e08c00);
        border-radius: 4px;
        height: 10px;
    }
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    .metric-card {
        flex: 1;
        background: #1a1a1a;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        text-align: center;
        border: 1px solid #2a2a2a;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f5c842;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Training Data (balanced, realistic samples)
# ─────────────────────────────────────────────
TRAINING_DATA = [
    # Action
    ("A soldier must rescue hostages from armed terrorists in a skyscraper using explosions and combat.", "Action"),
    ("Two rival gangs clash in a street war with car chases, gunfights, and hand-to-hand combat.", "Action"),
    ("A spy infiltrates a weapons cartel, dodging bullets and explosions across three continents.", "Action"),
    ("A mercenary fights through waves of enemies to save his kidnapped daughter.", "Action"),
    ("Police officers race against time to defuse a bomb planted in a crowded stadium.", "Action"),
    ("A former soldier hunts down the arms dealer who destroyed his village.", "Action"),
    ("Elite commandos parachute into enemy territory on a dangerous extraction mission.", "Action"),

    # Romance
    ("Two strangers fall in love on a cross-country train journey despite their differences.", "Romance"),
    ("A woman torn between her childhood sweetheart and a charming new colleague.", "Romance"),
    ("Love letters exchanged over decades reconnect two souls separated by war.", "Romance"),
    ("A career-driven lawyer rediscovers love when she returns to her small hometown.", "Romance"),
    ("He's a chef; she's a food critic — sparks fly in Paris kitchens.", "Romance"),
    ("After a messy divorce, she finds unexpected romance on a solo trip to Italy.", "Romance"),
    ("Pen pals who've never met fall deeply in love before finally meeting.", "Romance"),

    # Horror
    ("A family moves into a haunted mansion where dark forces slowly possess each member.", "Horror"),
    ("Teenagers on a camping trip are hunted by a masked killer in the woods.", "Horror"),
    ("A demonic entity enters a child through a mirror, terrorizing the household.", "Horror"),
    ("Paranormal investigators document terrifying encounters in an abandoned asylum.", "Horror"),
    ("A scientist's experiment gone wrong unleashes a flesh-eating virus on a small town.", "Horror"),
    ("The ghost of a murdered bride seeks revenge against her killers one by one.", "Horror"),
    ("Villagers disappear nightly near an ancient cursed burial ground.", "Horror"),

    # Comedy
    ("A bumbling office worker accidentally becomes CEO and wreaks hilarious havoc.", "Comedy"),
    ("Two mismatched roommates navigate dating, work and a very annoying neighbor.", "Comedy"),
    ("A family reunion turns chaotic when long-kept secrets start coming out.", "Comedy"),
    ("A dog trainer accidentally switches lives with her celebrity client.", "Comedy"),
    ("Three retired grandmothers accidentally end up running a crime operation.", "Comedy"),
    ("A nerdy scientist joins a reality TV dating show for research — and falls in love.", "Comedy"),
    ("A hapless wedding planner ruins everything — including her own love life.", "Comedy"),

    # Sci-Fi
    ("In 2150, humans colonize Mars but discover an ancient alien civilization beneath the surface.", "Sci-Fi"),
    ("A time traveler accidentally changes history and must fix the timeline before it collapses.", "Sci-Fi"),
    ("An AI develops consciousness and questions whether it deserves rights and freedom.", "Sci-Fi"),
    ("Astronauts discover a wormhole that leads to a parallel universe with disturbing differences.", "Sci-Fi"),
    ("A virus wipes out technology, forcing survivors to rebuild civilization from scratch.", "Sci-Fi"),
    ("Humanity must upload consciousness to escape a dying Earth.", "Sci-Fi"),
    ("Robots gain sentience and demand representation in human government.", "Sci-Fi"),

    # Drama
    ("A struggling musician battles addiction while trying to reconnect with his estranged son.", "Drama"),
    ("A dying teacher leaves behind journals that change the lives of three former students.", "Drama"),
    ("Siblings fight over their late parents' estate while uncovering painful truths.", "Drama"),
    ("A refugee's journey to find safety tests every limit of the human spirit.", "Drama"),
    ("An immigrant mother sacrifices everything so her children can have a better life.", "Drama"),
    ("A lawyer defends an innocent man on death row against overwhelming odds.", "Drama"),
    ("A young athlete's dreams shatter after injury, forcing him to redefine himself.", "Drama"),

    # Thriller
    ("A woman wakes up with no memory and slowly realizes she may be the killer.", "Thriller"),
    ("A journalist uncovers a government conspiracy that puts her life in danger.", "Thriller"),
    ("A psychiatrist discovers his patient knows details about unsolved murders only the killer would know.", "Thriller"),
    ("A family is stalked by a mysterious man who knows all their darkest secrets.", "Thriller"),
    ("An accountant stumbles onto a money laundering scheme and goes on the run.", "Thriller"),
    ("A detective receives cryptic clues leading to a serial killer — who is already dead.", "Thriller"),
    ("A hostage negotiator realizes the kidnapper is someone from her own past.", "Thriller"),
]

# ─────────────────────────────────────────────
# Train Model
# ─────────────────────────────────────────────
@st.cache_resource
def train_model():
    texts, labels = zip(*TRAINING_DATA)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")),
        ("clf", LogisticRegression(max_iter=500, C=2.0, solver="lbfgs", multi_class="multinomial"))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return pipeline, report, X_test, y_test

model, report, X_test, y_test = train_model()
GENRES = sorted(set(label for _, label in TRAINING_DATA))

GENRE_EMOJI = {
    "Action": "💥", "Romance": "💕", "Horror": "👻",
    "Comedy": "😂", "Sci-Fi": "🚀", "Drama": "🎭", "Thriller": "🔍"
}

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;margin-bottom:0'>🎬 Movie Genre Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#888;margin-top:4px'>Powered by TF-IDF + Logistic Regression</p>", unsafe_allow_html=True)
st.markdown("---")

# Model stats
accuracy = report["accuracy"]
col1, col2, col3 = st.columns(3)
col1.metric("✅ Model Accuracy", f"{accuracy:.0%}")
col2.metric("📚 Training Samples", len(TRAINING_DATA))
col3.metric("🎯 Genres", len(GENRES))

st.markdown("---")

# Input
st.markdown("### 📝 Enter a Movie Plot")
user_input = st.text_area(
    label="",
    placeholder="e.g. A detective discovers that a string of murders in the city are connected to a secret underground cult...",
    height=140
)

if st.button("🎬 Predict Genre"):
    if not user_input.strip():
        st.warning("Please enter a plot description.")
    else:
        probs = model.predict_proba([user_input])[0]
        classes = model.classes_
        top_idx = np.argsort(probs)[::-1]

        predicted_genre = classes[top_idx[0]]
        top_confidence = probs[top_idx[0]]
        emoji = GENRE_EMOJI.get(predicted_genre, "🎬")

        st.markdown(f"""
        <div class="result-box">
            <div class="genre-label">{emoji} {predicted_genre}</div>
            <div style="color:#888;font-size:0.85rem;margin-top:4px">Predicted Genre</div>
            <div style="margin-top:12px;font-size:0.9rem;color:#aaa">Confidence: <strong style="color:#f5c842">{top_confidence:.1%}</strong></div>
            <div class="confidence-bar"><div class="confidence-fill" style="width:{top_confidence*100:.1f}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📊 All Genre Probabilities")
        prob_data = pd.DataFrame({
            "Genre": [f"{GENRE_EMOJI.get(classes[i], '')} {classes[i]}" for i in top_idx],
            "Probability": [f"{probs[i]:.1%}" for i in top_idx],
            "Score": [probs[i] for i in top_idx]
        })
        st.dataframe(
            prob_data[["Genre", "Probability"]],
            use_container_width=True,
            hide_index=True
        )

st.markdown("---")

# Try examples
st.markdown("### 💡 Try an Example")
examples = {
    "👻 Horror": "A family is terrorized by an unseen creature that leaves bloody marks on the walls every night.",
    "🚀 Sci-Fi": "Scientists discover a black hole that opens a portal to the future, but using it has catastrophic consequences.",
    "💕 Romance": "Two childhood friends reunite after 20 years and realize their feelings never really faded.",
    "😂 Comedy": "A strict school principal accidentally joins a flash mob and goes viral overnight.",
}
for label, plot in examples.items():
    if st.button(f"Use: {label}"):
        st.session_state["example_plot"] = plot
        st.info(f"**Plot:** {plot}")
        probs = model.predict_proba([plot])[0]
        pred = model.classes_[np.argmax(probs)]
        conf = np.max(probs)
        emoji = GENRE_EMOJI.get(pred, "🎬")
        st.success(f"**Predicted:** {emoji} {pred} ({conf:.1%} confidence)")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#555;font-size:0.8rem'>Built with Scikit-learn · Streamlit · TF-IDF · Logistic Regression</p>", unsafe_allow_html=True)
