# randon number generation like otp generation
# import random 

# a = 1000
# b= 10000
# otp = random.randint(a,b+1)
# print(otp)


# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from urllib.parse import quote_plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ----------------------------
# Page config and Theme
# ----------------------------
st.set_page_config(
    page_title="CineSense • Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Optional: Streamlit theme (light/dark handled by settings). Add subtle CSS for cards.
CARD_CSS = """
<style>
.card {
  border-radius: 14px;
  border: 1px solid #e6e6e6;
  padding: 14px;
  margin-bottom: 14px;
  background: var(--background-color);
  transition: box-shadow 0.2s ease-in-out;
}
.card:hover { box-shadow: 0 6px 24px rgba(0,0,0,0.06); }
.poster {
  width: 100%;
  border-radius: 10px;
  object-fit: cover;
}
.title {
  font-weight: 700;
  font-size: 1.05rem;
  margin: 6px 0 4px 0;
}
.meta {
  color: #6b7280;
  font-size: 0.9rem;
  margin-bottom: 8px;
}
.link a {
  text-decoration: none;
  color: #2563eb;
  font-weight: 600;
}
.badge {
  display: inline-block;
  background: #eef2ff;
  color: #3730a3;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 0.75rem;
  margin-right: 6px;
}
.section-label {
  color: #6b7280;
  font-size: 0.85rem;
  margin-bottom: 6px;
  text-transform: uppercase;
  letter-spacing: .06em;
}
hr.soft {
  border: none;
  border-top: 1px solid #eeeeee;
  margin: 8px 0 12px;
}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

# ----------------------------
# Demo dataset (replace with your own)
# Columns: id, title, overview, year, poster_url, info_url, genres
# ----------------------------
DATA = [
    {
        "id": 1,
        "title": "Inception",
        "overview": "A thief who steals corporate secrets through dream-sharing technology is given a chance at redemption by planting an idea into a target's subconscious.",
        "year": 2010,
        "poster_url": "https://image.tmdb.org/t/p/w342/qmDpIHrmpJINaRKAfWQfftjCdyi.jpg",
        "info_url": "https://www.themoviedb.org/movie/27205-inception",
        "genres": "Action, Sci-Fi, Thriller"
    },
    {
        "id": 2,
        "title": "Interstellar",
        "overview": "A team of explorers travels through a wormhole in space in an attempt to ensure humanity's survival.",
        "year": 2014,
        "poster_url": "https://image.tmdb.org/t/p/w342/rAiYTfKGqDCRIIqo664sY9XZIvQ.jpg",
        "info_url": "https://www.themoviedb.org/movie/157336-interstellar",
        "genres": "Adventure, Drama, Sci-Fi"
    },
    {
        "id": 3,
        "title": "The Dark Knight",
        "overview": "Batman faces the Joker, a criminal mastermind, who plunges Gotham into chaos, testing the limits of the caped crusader.",
        "year": 2008,
        "poster_url": "https://image.tmdb.org/t/p/w342/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
        "info_url": "https://www.themoviedb.org/movie/155-the-dark-knight",
        "genres": "Action, Crime, Drama"
    },
    {
        "id": 4,
        "title": "Arrival",
        "overview": "A linguist is recruited to communicate with extraterrestrial visitors and unravels a profound mystery about time and language.",
        "year": 2016,
        "poster_url": "https://image.tmdb.org/t/p/w342/x2FJsf1ElAgr63Y3PNPtJrcmpoe.jpg",
        "info_url": "https://www.themoviedb.org/movie/329865-arrival",
        "genres": "Drama, Sci-Fi, Mystery"
    },
    {
        "id": 5,
        "title": "The Social Network",
        "overview": "Harvard student Mark Zuckerberg creates the social networking site that would become Facebook, facing legal and personal complications.",
        "year": 2010,
        "poster_url": "https://image.tmdb.org/t/p/w342/n0ybibhJtQ5icDqTp8eRytcIHJx.jpg",
        "info_url": "https://www.themoviedb.org/movie/37799-the-social-network",
        "genres": "Drama, Biography"
    },
    {
        "id": 6,
        "title": "Blade Runner 2049",
        "overview": "A young blade runner's discovery leads him to track down former blade runner Rick Deckard, missing for decades.",
        "year": 2017,
        "poster_url": "https://image.tmdb.org/t/p/w342/aMpyrCizvSgKHRj7XdzF7z2PWBz.jpg",
        "info_url": "https://www.themoviedb.org/movie/335984-blade-runner-2049",
        "genres": "Sci-Fi, Mystery, Drama"
    },
    {
        "id": 7,
        "title": "La La Land",
        "overview": "A jazz musician and an aspiring actress fall in love while pursuing their dreams in Los Angeles.",
        "year": 2016,
        "poster_url": "https://image.tmdb.org/t/p/w342/uDO8zWDhfWwoFdKS4fzkUJt0Rf0.jpg",
        "info_url": "https://www.themoviedb.org/movie/313369-la-la-land",
        "genres": "Comedy, Drama, Romance"
    },
    {
        "id": 8,
        "title": "Mad Max: Fury Road",
        "overview": "In a post-apocalyptic wasteland, a drifter and a warrior rebel against a tyrannical ruler in a high-octane road war.",
        "year": 2015,
        "poster_url": "https://image.tmdb.org/t/p/w342/kqjL17yufvn9OVLyXYpvtyrFfak.jpg",
        "info_url": "https://www.themoviedb.org/movie/76341-mad-max-fury-road",
        "genres": "Action, Adventure, Sci-Fi"
    },
    {
        "id": 9,
        "title": "Whiplash",
        "overview": "A young drummer enrolls at a cut-throat music conservatory where his dreams are mentored by an instructor who will stop at nothing.",
        "year": 2014,
        "poster_url": "https://image.tmdb.org/t/p/w342/lIv1QinFqz4dlp5U4lQ6HaiskOZ.jpg",
        "info_url": "https://www.themoviedb.org/movie/244786-whiplash",
        "genres": "Drama, Music"
    },
    {
        "id": 10,
        "title": "Her",
        "overview": "A lonely writer develops an unlikely relationship with an operating system designed to meet his every need.",
        "year": 2013,
        "poster_url": "https://image.tmdb.org/t/p/w342/pH7ZzH2YFQhC1Kqz3PaY5E9OQgm.jpg",
        "info_url": "https://www.themoviedb.org/movie/152601-her",
        "genres": "Romance, Sci-Fi, Drama"
    },
]

df = pd.DataFrame(DATA)

# ----------------------------
# Build a simple content-based model
# ----------------------------
@st.cache_resource(show_spinner=False)
def build_model(frame: pd.DataFrame):
    corpus = (frame["overview"].fillna("") + " " + frame["genres"].fillna("")).values
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    tfidf = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf

vectorizer, tfidf = build_model(df)

def recommend_from_text(query: str, top_k: int = 6):
    if not query.strip():
        return []
    q_vec = vectorizer.transform([query])
    sims = linear_kernel(q_vec, tfidf).ravel()
    idx = np.argsort(-sims)[:top_k]
    results = df.iloc[idx].copy()
    results["score"] = sims[idx]
    return results

# ----------------------------
# Header
# ----------------------------
left, right = st.columns([0.75, 0.25])
with left:
    st.title("🎬 CineSense")
    st.caption("Describe a movie vibe or plot, get instant recommendations.")
with right:
    st.markdown("<div class='section-label'>Theme</div>", unsafe_allow_html=True)
    theme_choice = st.radio("", ["Auto", "Light", "Dark"], horizontal=True, label_visibility="collapsed")
    # Tip: Streamlit theme is set in settings or config.toml.
    # We provide a simple suggestion box.
    if theme_choice != "Auto":
        st.info("Tip: Set theme in Settings → Theme for a persistent choice.", icon="💡")

st.markdown("----")

# ----------------------------
# Sidebar: Filters
# ----------------------------
with st.sidebar:
    st.header("Filters")
    topk = st.slider("Results to show", min_value=3, max_value=12, value=6, step=1)
    gen_pref = st.multiselect(
        "Prefer genres",
        options=sorted({g.strip() for row in df.genres for g in row.split(",")}),
        default=[]
    )
    st.caption("Genre preference softly re-ranks results if matched.")

def soft_rerank_by_genre(results: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    if not preferred or results.empty:
        return results
    pref_set = {g.lower() for g in preferred}
    bonus = []
    for row in results.itertuples():
        gset = {g.strip().lower() for g in row.genres.split(",")}
        overlap = len(pref_set & gset)
        bonus.append(0.05 * overlap)  # small, additive bonus
    results = results.copy()
    results["score"] = results["score"].values + np.array(bonus)
    results = results.sort_values("score", ascending=False)
    return results

# ----------------------------
# Input area
# ----------------------------
prompt = st.text_area(
    "What are you in the mood for?",
    placeholder="e.g., cerebral sci-fi about time and language, emotional, slow-burn",
    height=80
)

col_a, col_b = st.columns([0.2, 0.8])
with col_a:
    search = st.button("Recommend", use_container_width=True)

with col_b:
    st.caption("Describe themes, vibes, pace, setting, or similar titles. Example: “like Interstellar but more emotional, with space exploration and family drama.”")

# ----------------------------
# Run recommendation
# ----------------------------
if search:
    results = recommend_from_text(prompt, top_k=topk)
    results = soft_rerank_by_genre(results, gen_pref)

    if len(results) == 0:
        st.warning("No results. Try adding more detail or different keywords.")
    else:
        # Grid layout
        cols = st.columns(3)
        for i, row in enumerate(results.itertuples()):
            with cols[i % 3]:
                poster = row.poster_url or ""
                title = row.title
                year = row.year
                overview = row.overview
                info_url = row.info_url
                genres = row.genres

                # Fallback poster if needed
                if not poster:
                    # quick static fallback; alternatively, query an image API
                    poster = "https://via.placeholder.com/342x513.png?text=No+Image"

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.image(poster, use_container_width=True, caption=None)
                st.markdown(f"<div class='title'>{title} ({year})</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='meta'>{genres}</div>", unsafe_allow_html=True)

                # Overview trimmed
                trimmed = overview if len(overview) < 220 else overview[:220].rsplit(" ", 1)[0] + "..."
                st.write(trimmed)

                # Link and badges
                st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
                st.markdown(f"<div class='link'>🔗 <a href='{info_url}' target='_blank' rel='noopener'>More details</a></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Search helper (optional)
# ----------------------------
with st.expander("No luck? Try these prompts"):
    st.write("- Thought-provoking sci-fi about memory and identity, visually striking, slow pace")
    st.write("- Gritty action with moral dilemmas, grounded, urban setting")
    st.write("- Music-driven drama with intense mentorship and obsession")
    st.write("- Offbeat romance with technology themes, intimate tone")

# ----------------------------
# Footer
# ----------------------------
st.markdown("----")
st.caption("Demo content-based recommendations using TF-IDF over overviews + genres. Replace dataset with your own or TMDB/IMDb exports for better results.")
