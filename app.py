import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- page configuration ---
st.set_page_config(
    page_title="Ultimate Movie Discovery Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- custom CSS for premium glassmorphism styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,500;0,700;1,400&display=swap');

/* Global Styles */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: linear-gradient(135deg, #0a0714 0%, #120e24 50%, #030107 100%) !important;
    color: #e2e0ff !important;
    font-family: 'Montserrat', sans-serif !important;
}

/* Hide Streamlit default decorations */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #0a0714;
}
::-webkit-scrollbar-thumb {
    background: #7f00ff;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #ff007f;
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: #0b0818 !important;
    border-right: 1px solid rgba(127, 0, 255, 0.15) !important;
    padding-top: 20px;
}

/* Heading Gradients */
.glowing-title {
    font-family: 'Montserrat', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    background: linear-gradient(90deg, #ff007f 0%, #7f00ff 50%, #00f0ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
    text-shadow: 0px 0px 30px rgba(127, 0, 255, 0.2);
}

.glowing-subtitle {
    font-size: 1.1rem;
    color: #b0aeff;
    margin-bottom: 30px;
    font-weight: 400;
}

/* Glassmorphism Card styling */
.movie-card {
    position: relative;
    border-radius: 16px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

.movie-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 30px rgba(127, 0, 255, 0.25);
    border: 1px solid rgba(127, 0, 255, 0.35);
}

/* Movie Rating / Year Badges */
.movie-badge {
    position: absolute;
    top: 12px;
    right: 12px;
    background: rgba(10, 7, 20, 0.85);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: #ffc107;
    padding: 5px 10px;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 700;
    z-index: 2;
    display: flex;
    align-items: center;
    gap: 4px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}

.genre-pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 8px;
    background: rgba(127, 0, 255, 0.12);
    border: 1px solid rgba(127, 0, 255, 0.25);
    color: #dfc5ff;
    font-size: 0.72rem;
    margin-right: 6px;
    margin-bottom: 6px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Premium Buttons overrides */
.stButton>button {
    background: linear-gradient(90deg, #7f00ff 0%, #ff007f 100%) !important;
    color: #ffffff !important;
    border: none !important;
    padding: 10px 24px !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: 'Montserrat', sans-serif !important;
    letter-spacing: 0.5px;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(127, 0, 255, 0.3) !important;
}

.stButton>button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 6px 20px rgba(255, 0, 127, 0.5) !important;
}

/* Input Fields styling overrides */
div[data-baseweb="select"], div[data-baseweb="input"] {
    background-color: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
}

/* Text overrides */
.stMarkdown p, .stMarkdown li {
    font-weight: 400;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# --- system constants ---
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# --- fallbacks ---
FALLBACK_POSTER = "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=500&auto=format&fit=crop"
FALLBACK_BACKDROP = "https://images.unsplash.com/photo-1536440136628-849c177e76a1?q=80&w=1280&auto=format&fit=crop"
FALLBACK_PROFILE = "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?q=80&w=150&auto=format&fit=crop"

# --- 1. curated dataset of 50 blockbusters ---
MOVIES_DATA = [
    {"movie_id": 19995, "title": "Avatar", "tags": "Action Adventure Fantasy Sci-Fi space sky war alien pandora 3d epic blue james cameron sam worthington zoe saldana sigourney weaver"},
    {"movie_id": 24428, "title": "The Avengers", "tags": "Action Adventure Sci-Fi marvel superheroes comic book team avengers iron man captain america thor hulk black widow hawkeye loki robert downey jr chris evans"},
    {"movie_id": 299536, "title": "Avengers: Infinity War", "tags": "Action Adventure Sci-Fi marvel superheroes thanos infinity stones war epic snap robert downey jr chris hemsworth mark ruffalo"},
    {"movie_id": 157336, "title": "Interstellar", "tags": "Adventure Drama Sci-Fi space travel wormhole black hole time travel future earth space exploration gravity christopher nolan matthew mcconaughey anne hathaway"},
    {"movie_id": 27205, "title": "Inception", "tags": "Action Sci-Fi Thriller dream heist reality mind-bending subconscious christopher nolan leonardo dicaprio joseph gordon-levitt elliot page"},
    {"movie_id": 68718, "title": "Django Unchained", "tags": "Western Drama Action bounty hunter slavery revenge classic tarantino jamie foxx christoph waltz leonardo dicaprio"},
    {"movie_id": 550, "title": "Fight Club", "tags": "Drama Action Thriller split personality psychological underground club rebellion brad pitt edward norton helena bonham carter david fincher"},
    {"movie_id": 11, "title": "Star Wars: A New Hope", "tags": "Action Adventure Sci-Fi space opera jedi empire space force luke skywalker han solo princess leia darth vader classic george lucas mark hamill harrison ford"},
    {"movie_id": 122, "title": "The Lord of the Rings: The Return of the King", "tags": "Adventure Fantasy Action epic ring middle earth hobbit gandalf trilogies frodo sam aragorn golum peter jackson elijah wood viggo mortensen"},
    {"movie_id": 13, "title": "Forrest Gump", "tags": "Drama Romance Comedy running history life inspiration classic tom hanks robin wright"},
    {"movie_id": 155, "title": "The Dark Knight", "tags": "Action Crime Drama Thriller superhero batman joker gothic vigilante dc comics christopher nolan christian bale heath ledger gary oldman"},
    {"movie_id": 603, "title": "The Matrix", "tags": "Action Sci-Fi cyber hacking reality simulation martial arts bullet time classic keanu reeves laurence fishburne carrie-anne moss"},
    {"movie_id": 680, "title": "Pulp Fiction", "tags": "Thriller Crime drama non-linear narrative hitman boxing classic tarantino john travolta samuel l jackson uma thurman"},
    {"movie_id": 597, "title": "Titanic", "tags": "Drama Romance Historical ship disaster iceberg love story ocean leonardo dicaprio kate winslet james cameron"},
    {"movie_id": 278, "title": "The Shawshank Redemption", "tags": "Drama prison escape hope friendship justice classic morgan freeman tim robbins"},
    {"movie_id": 238, "title": "The Godfather", "tags": "Drama Crime mafia family loyalty godfather crime family classic marlon brando al pacino francis ford coppola"},
    {"movie_id": 98, "title": "Gladiator", "tags": "Action Adventure Drama rome gladiator revenge empire battle historic russell crowe joaquin phoenix ridley scott"},
    {"movie_id": 496243, "title": "Parasite", "tags": "Thriller Drama Comedy social satire class divide korea family manipulation bong joon ho song kang-ho"},
    {"movie_id": 244786, "title": "Whiplash", "tags": "Drama Music jazz drums obsession mentor ambition miles teller jk simmons damien chazelle"},
    {"movie_id": 313369, "title": "La La Land", "tags": "Romance Music Comedy drama jazz hollywood musical dreamers ryan gosling emma stone damien chazelle"},
    {"movie_id": 129, "title": "Spirited Away", "tags": "Animation Fantasy Adventure anime family spirit world magic classic hayao miyazaki"},
    {"movie_id": 324857, "title": "Spider-Man: Into the Spider-Verse", "tags": "Animation Action Adventure superhero multiverse miles morales spider-man comic book comic style marvel"},
    {"movie_id": 335984, "title": "Blade Runner 2049", "tags": "Sci-Fi Thriller Drama cyberpunk replica future neon mystery denis villeneuve ryan gosling harrison ford"},
    {"movie_id": 76341, "title": "Mad Max: Fury Road", "tags": "Action Adventure Sci-Fi post-apocalyptic desert car chase survival action-packed tom hardy charlize theron"},
    {"movie_id": 438631, "title": "Dune", "tags": "Sci-Fi Adventure space dessert spice empire messiah book adaptation denis villeneuve timothee chalamet zendaya"},
    {"movie_id": 419430, "title": "Get Out", "tags": "Horror Thriller Mystery racism psychological social thriller jordan peele daniel kaluya"},
    {"movie_id": 546554, "title": "Knives Out", "tags": "Comedy Mystery Thriller whodunnit detective murder family drama daniel craig chris evans ana de armas"},
    {"movie_id": 194, "title": "The Truman Show", "tags": "Drama Comedy reality tv simulation freedom media satire jim carrey ed harris"},
    {"movie_id": 329865, "title": "Arrival", "tags": "Sci-Fi Drama Mystery alien language communication time linguistics denis villeneuve amy adams jeremy renner"},
    {"movie_id": 11324, "title": "Shutter Island", "tags": "Mystery Thriller Drama psychological asylum detective twist ending martin scorsese leonardo dicaprio"},
    {"movie_id": 807, "title": "Se7en", "tags": "Mystery Thriller Crime serial killer detective dark gritty seven deadly sins david fincher brad pitt morgan freeman"},
    {"movie_id": 274, "title": "The Silence of the Lambs", "tags": "Thriller Horror Crime fbi psychological hannibal lecter serial killer classic anthony hopkins jodie foster"},
    {"movie_id": 38, "title": "Eternal Sunshine of the Spotless Mind", "tags": "Romance Sci-Fi Drama memory erasure heartbreak relationship mind-bending jim carrey kate winslet"},
    {"movie_id": 12445, "title": "Harry Potter and the Deathly Hallows: Part 2", "tags": "Fantasy Adventure Action magic wizard voldemort hogwarts battle epic daniel radcliffe emma watson rupert grint"},
    {"movie_id": 634649, "title": "Spider-Man: No Way Home", "tags": "Action Adventure Sci-Fi superheroes multiverse marvel comic book spider-man tom holland zendaya benedict cumberbatch"},
    {"movie_id": 872585, "title": "Oppenheimer", "tags": "Drama History biography atomic bomb war physicist christopher nolan cillian murphy robert downey jr emily blunt"},
    {"movie_id": 346698, "title": "Barbie", "tags": "Comedy Fantasy dolls real world feminism existential crisis colorful margot robbie ryan gosling greta gerwig"},
    {"movie_id": 475554, "title": "Joker", "tags": "Drama Crime Thriller mental illness madness clown origin story dc comics Joaquin Phoenix"},
    {"movie_id": 106646, "title": "The Wolf of Wall Street", "tags": "Comedy Drama Crime finance stocks excess drugs martin scorsese leonardo dicaprio margot robbie"},
    {"movie_id": 640, "title": "Catch Me If You Can", "tags": "Drama Crime biography con artist fbi chase tom hanks leonardo dicaprio martin scorsese"},
    {"movie_id": 105, "title": "Back to the Future", "tags": "Adventure Comedy Sci-Fi time travel 80s classic marty mcfly doc brown robert zemeckis michael j fox"},
    {"movie_id": 329, "title": "Jurassic Park", "tags": "Adventure Sci-Fi dinosaurs theme park disaster classic steven spielberg sam neill laura dern jeff goldblum"},
    {"movie_id": 348, "title": "Alien", "tags": "Horror Sci-Fi space spaceship alien creature survival classic ridley scott sigourney weaver"},
    {"movie_id": 694, "title": "The Shining", "tags": "Horror Thriller hotel isolation madness supernatural classic stanley kubrick jack nicholson"},
    {"movie_id": 150540, "title": "Inside Out", "tags": "Animation Comedy Drama family emotions mind childhood disney pixar"},
    {"movie_id": 354912, "title": "Coco", "tags": "Animation Family Music fantasy day of the dead mexico guitar skeleton disney pixar"},
    {"movie_id": 8587, "title": "The Lion King", "tags": "Animation Drama Family musical africa animals lion king hamlet classic disney"},
    {"movie_id": 10681, "title": "WALL-E", "tags": "Animation Sci-Fi Family robot space future earth environment love story pixar disney"},
    {"movie_id": 2062, "title": "Ratatouille", "tags": "Animation Comedy Family paris chef cooking rat mouse disney pixar"},
    {"movie_id": 14160, "title": "Up", "tags": "Animation Comedy Adventure Family house balloons adventure elderly wilderness pixar disney"}
]

df = pd.DataFrame(MOVIES_DATA)

# --- 2. machine learning setup (content-based vectors) ---
@st.cache_resource
def get_ml_vectors(dataframe):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(dataframe['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return cv, vectors, similarity

cv, vectors, similarity_matrix = get_ml_vectors(df)

# --- 3. cached TMDB API functions ---
@st.cache_data(ttl=86400)
def get_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US&append_to_response=videos,credits"
    try:
        res = requests.get(url, headers=HEADERS, timeout=5)
        if res.status_code == 200:
            return res.json()
    except:
        pass
    return None

@st.cache_data(ttl=3600)
def search_tmdb_movies(query):
    if not query.strip():
        return []
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}&language=en-US"
    try:
        res = requests.get(url, headers=HEADERS, timeout=5)
        if res.status_code == 200:
            return res.json().get('results', [])
    except:
        pass
    return []

@st.cache_data(ttl=1800)
def get_tmdb_recommendations(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/recommendations?api_key={TMDB_API_KEY}&language=en-US"
    try:
        res = requests.get(url, headers=HEADERS, timeout=5)
        if res.status_code == 200:
            return res.json().get('results', [])
    except:
        pass
    return []

# --- helper functions to construct URLs ---
def get_poster_url(path):
    if path:
        return f"https://image.tmdb.org/t/p/w500{path}"
    return FALLBACK_POSTER

def get_backdrop_url(path):
    if path:
        return f"https://image.tmdb.org/t/p/w1280{path}"
    return FALLBACK_BACKDROP

def get_profile_url(path):
    if path:
        return f"https://image.tmdb.org/t/p/w185{path}"
    return FALLBACK_PROFILE

# --- 4. session state initialization ---
if 'selected_movie_id' not in st.session_state:
    st.session_state.selected_movie_id = 157336  # Default: Interstellar

if 'global_selected_movie_id' not in st.session_state:
    st.session_state.global_selected_movie_id = 27205  # Default: Inception

if 'global_search_results' not in st.session_state:
    st.session_state.global_search_results = []

# Dynamic Mapping to track correct selectbox positions
movie_titles = list(df['title'].values)
current_movie_title = df[df['movie_id'] == st.session_state.selected_movie_id].iloc[0]['title']
selectbox_default_index = movie_titles.index(current_movie_title)

def on_selectbox_change():
    chosen_title = st.session_state.curated_selectbox_key
    chosen_id = df[df['title'] == chosen_title].iloc[0]['movie_id']
    st.session_state.selected_movie_id = chosen_id

# --- 5. sidebar recommendation settings (filters) ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=300&auto=format&fit=crop", use_container_width=True)
    st.markdown("<h3 style='color:#fff; margin-top:15px; font-weight:700;'>🔧 Recommendation Controls</h3>", unsafe_allow_html=True)
    
    sidebar_min_rating = st.slider(
        "Minimum Rating", 
        min_value=0.0, 
        max_value=10.0, 
        value=6.0, 
        step=0.5,
        help="Filter suggestions above this TMDB rating score"
    )
    
    sidebar_years = st.slider(
        "Release Year Range",
        min_value=1970,
        max_value=2026,
        value=(1980, 2026),
        step=1,
        help="Filter suggestions published within this year range"
    )
    sidebar_start_year, sidebar_end_year = sidebar_years
    
    genres_list = ["All", "Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "Thriller", "Western"]
    sidebar_genre = st.selectbox(
        "Filter by Genre",
        options=genres_list,
        index=0,
        help="Only display recommended movies from this genre category"
    )
    
    st.markdown("<hr style='border-color: rgba(127,0,255,0.25);'>", unsafe_allow_html=True)
    st.markdown("Made with ❤️ using TMDB API & Scikit-Learn Similarity Engine.")

# --- 6. header titles ---
st.markdown("<div class='glowing-title'>🎬 Ultimate Movie Discovery Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='glowing-subtitle'>An advanced hybrid recommendation dashboard with live trailers, cast profiles, filters & mood search</div>", unsafe_allow_html=True)

# --- 7. tabs navigation ---
tabs = st.tabs(["🍿 Curated Discovery", "🌐 Global Search Graph", "🧠 Mood Recommender"])

# ================= TAB 1: CURATED DISCOVERY =================
with tabs[0]:
    # Selectbox dynamic index structure to resolve layout dependencies
    selected_title = st.selectbox(
        "Select a movie to explore similarities:",
        options=movie_titles,
        index=selectbox_default_index,
        key="curated_selectbox_key",
        on_change=on_selectbox_change
    )

    # Load Details from TMDB
    details = get_movie_details(st.session_state.selected_movie_id)
    if details:
        # Render cinematic hero details card
        backdrop = get_backdrop_url(details.get('backdrop_path'))
        poster = get_poster_url(details.get('poster_path'))
        title = details.get('title')
        tagline = details.get('tagline', '')
        rating = round(details.get('vote_average', 0.0), 1)
        r_date = details.get('release_date', '')
        year = r_date.split('-')[0] if r_date else 'N/A'
        runtime = details.get('runtime', 0)
        genres = [g['name'] for g in details.get('genres', [])]
        overview = details.get('overview', 'No synopsis available.')
        
        # Build clean formatted genre pills
        genre_pills_html = "".join([f'<span class="genre-pill">{g}</span>' for g in genres])
        tagline_html = f'<p style="font-style: italic; color: #00f0ff; margin: 8px 0 16px 0; font-size: 1.1rem; font-family: Montserrat, sans-serif;">"{tagline}"</p>' if tagline else '<div style="margin-top: 15px;"></div>'
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(180deg, rgba(10, 7, 20, 0.4) 0%, rgba(10, 7, 20, 0.96) 100%), url('{backdrop}');
            background-size: cover;
            background-position: center;
            border-radius: 16px;
            padding: 30px;
            border: 1px solid rgba(127, 0, 255, 0.15);
            margin-bottom: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        ">
            <div style="display: flex; flex-direction: row; flex-wrap: wrap; gap: 30px; align-items: flex-start;">
                <img src="{poster}" style="width: 220px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.6); border: 1px solid rgba(255,255,255,0.15);" />
                <div style="flex: 1; min-width: 300px; color: #fff;">
                    <h1 style="margin: 0; font-size: 2.4rem; font-weight: 800; background: linear-gradient(90deg, #ff007f, #7f00ff, #00f0ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{title}</h1>
                    {tagline_html}
                    <div style="display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 20px;">
                        <span style="background: rgba(255, 193, 7, 0.15); border: 1px solid #ffc107; color: #ffc107; padding: 5px 12px; border-radius: 8px; font-weight: 700; display: inline-flex; align-items: center; gap: 4px; font-size: 0.85rem;">⭐ {rating}/10</span>
                        <span style="background: rgba(255, 255, 255, 0.05); padding: 5px 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.15); font-size: 0.85rem;">📅 {year}</span>
                        <span style="background: rgba(255, 255, 255, 0.05); padding: 5px 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.15); font-size: 0.85rem;">⏱️ {runtime} min</span>
                    </div>
                    <div style="margin-bottom: 20px;">
                        {genre_pills_html}
                    </div>
                    <h3 style="font-size: 1.25rem; color: #7f00ff; margin: 20px 0 8px 0; font-weight: 700;">Synopsis</h3>
                    <p style="font-size: 0.95rem; line-height: 1.6; color: #ddd; margin: 0 0 20px 0;">{overview}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Cast Section - FIXED HTML Wrapper Integration
        cast = details.get('credits', {}).get('cast', [])[:5]
        if cast:
            st.markdown("<h3 style='margin: 0 0 15px 0; font-size: 1.3rem; font-weight: 700; color:#fff;'>👥 Starring Cast</h3>", unsafe_allow_html=True)
            cast_cols = st.columns(len(cast))
            for actor_idx, actor in enumerate(cast):
                with cast_cols[actor_idx]:
                    act_name = actor.get('name')
                    act_char = actor.get('character')
                    act_profile = get_profile_url(actor.get('profile_path'))
                    
                    st.markdown(f"""
                    <div style='display: flex; align-items: center; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.15); height: 100%;'>
                        <img src='{act_profile}' style='width: 45px; height: 45px; border-radius: 50%; object-fit: cover; border: 2px solid #7f00ff; flex-shrink: 0;' />
                        <div style='margin-left: 10px; overflow: hidden;'>
                            <div style='font-size: 0.85rem; font-weight: 700; color: #fff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>{act_name}</div>
                            <div style='font-size: 0.72rem; color: #b0aeff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>{act_char}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)
            
        # Video trailer
        videos = details.get('videos', {}).get('results', [])
        trailer_key = None
        for v in videos:
            if v.get('site') == 'YouTube' and v.get('type') in ['Trailer', 'Teaser']:
                trailer_key = v.get('key')
                break
        if trailer_key:
            st.markdown("<h3 style='margin: 0 0 10px 0; font-size: 1.3rem; font-weight: 700; color:#fff;'>🎥 Video Trailer</h3>", unsafe_allow_html=True)
            st.video(f"https://www.youtube.com/watch?v={trailer_key}")
            st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

        # Recommendation Generation using local ML cosine similarity
        movie_idx = df[df['movie_id'] == st.session_state.selected_movie_id].index[0]
        distances = similarity_matrix[movie_idx]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])

        # Filter movies
        recommended_items = []
        for idx, sim in movies_list:
            if idx == movie_idx:
                continue
            m_id = df.iloc[idx]['movie_id']
            m_title = df.iloc[idx]['title']
            
            # Retrieve cached details for filtering
            m_details = get_movie_details(m_id)
            if not m_details:
                continue
            
            # parse metadata
            r_date = m_details.get('release_date', '')
            r_year = int(r_date.split('-')[0]) if (r_date and r_date.split('-')[0].isdigit()) else 0
            r_rating = m_details.get('vote_average', 0.0)
            r_genres = [g['name'] for g in m_details.get('genres', [])]
            
            # check filter constraints
            if r_rating < sidebar_min_rating:
                continue
            if r_year < sidebar_start_year or r_year > sidebar_end_year:
                continue
            if sidebar_genre != "All" and sidebar_genre not in r_genres:
                continue
                
            recommended_items.append({
                'movie_id': m_id,
                'title': m_title,
                'rating': round(r_rating, 1),
                'year': r_year,
                'poster_url': get_poster_url(m_details.get('poster_path')),
                'runtime': m_details.get('runtime', 0),
                'genres': r_genres,
                'similarity': sim
            })
            
            if len(recommended_items) >= 6:
                break
                
        # Render Recommendations Grid
        st.markdown("<hr style='border-color: rgba(127,0,255,0.2); margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin: 0 0 20px 0; font-size: 1.6rem; font-weight: 700; color:#fff;'>🍿 Handpicked Similar Discoveries</h3>", unsafe_allow_html=True)
        
        if not recommended_items:
            st.info("No recommendations matching your current filters. Adjust your controls in the sidebar to reveal more choices.")
        else:
            cols = st.columns(3)
            for i, item in enumerate(recommended_items):
                col_idx = i % 3
                with cols[col_idx]:
                    genres_html = " ".join([f'<span class="genre-pill">{g}</span>' for g in item['genres'][:2]])
                    card_html = f"""
                    <div class="movie-card" style="margin-bottom: 12px;">
                        <div class="movie-badge">⭐ {item['rating']}</div>
                        <img src="{item['poster_url']}" style="width: 100%; height: 320px; object-fit: cover; display: block;" />
                        <div style="padding: 15px;">
                            <h4 style="margin: 0 0 5px 0; font-size: 1.05rem; color: #fff; font-weight: 700; min-height: 48px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">{item['title']}</h4>
                            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #aaa; margin-top: 5px; margin-bottom: 8px;">
                                <span>📅 {item['year']}</span>
                                <span>⏱️ {item['runtime']}m</span>
                            </div>
                            <div style="min-height: 30px;">
                                {genres_html}
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
                    
                    # Core Session Fix: Triggering state modification on active runtime loops
                    if st.button("🔍 View Details", key=f"rec_detail_{item['movie_id']}", use_container_width=True):
                        st.session_state.selected_movie_id = item['movie_id']
                        st.rerun()

# ================= TAB 2: GLOBAL SEARCH GRAPH =================
with tabs[1]:
    st.markdown("<h3 style='margin: 0 0 10px 0; font-size: 1.4rem; font-weight: 700; color:#fff;'>🌐 Query the Complete TMDB Global Database</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.9rem; color:#aaa; margin-bottom:15px;'>Search for absolutely any movie (live TMDB API search) and get deep similarity mappings dynamically.</p>", unsafe_allow_html=True)
    
    search_col1, search_col2 = st.columns([5, 1])
    with search_col1:
        query_input = st.text_input("Enter Movie Name:", placeholder="e.g. Oppenheimer, Barbie, Spider-Man...", key="global_search_input", label_visibility="collapsed")
    with search_col2:
        search_trigger = st.button("Search", key="global_search_trigger", use_container_width=True)

    if search_trigger:
        if query_input:
            with st.spinner("Searching global databases..."):
                st.session_state.global_search_results = search_tmdb_movies(query_input)

    if st.session_state.global_search_results:
        st.markdown("<h3 style='margin: 20px 0 15px 0; font-size: 1.25rem; font-weight: 700; color:#fff;'>🔍 Search Matches</h3>", unsafe_allow_html=True)
        cols = st.columns(4)
        for idx, item in enumerate(st.session_state.global_search_results[:8]):
            col_idx = idx % 4
            with cols[col_idx]:
                r_id = item.get('id')
                r_title = item.get('title')
                r_poster = get_poster_url(item.get('poster_path'))
                r_date = item.get('release_date', '')
                r_year = r_date.split('-')[0] if r_date else 'N/A'
                r_rating = round(item.get('vote_average', 0.0), 1)
                
                card_html = f"""
                <div class="movie-card" style="margin-bottom: 12px;">
                    <div class="movie-badge">⭐ {r_rating}</div>
                    <img src="{r_poster}" style="width: 100%; height: 260px; object-fit: cover; display: block;" />
                    <div style="padding: 10px;">
                        <h5 style="margin: 0; font-size: 0.9rem; color: #fff; font-weight: 700; min-height: 40px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">{r_title} ({r_year})</h5>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button("🎯 Select Movie", key=f"sel_gl_{r_id}", use_container_width=True):
                    st.session_state.global_selected_movie_id = r_id
                    st.rerun()

    if st.session_state.global_selected_movie_id:
        g_details = get_movie_details(st.session_state.global_selected_movie_id)
        if g_details:
            st.markdown("<hr style='border-color: rgba(127,0,255,0.2); margin: 30px 0;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='margin: 0 0 15px 0; font-size: 1.5rem; font-weight: 700; color:#fff;'>🎬 Selected Movie Details</h3>", unsafe_allow_html=True)
            
            g_poster = get_poster_url(g_details.get('poster_path'))
            g_title = g_details.get('title')
            g_tagline = g_details.get('tagline', '')
            g_rating = round(g_details.get('vote_average', 0.0), 1)
            g_r_date = g_details.get('release_date', '')
            g_year = g_r_date.split('-')[0] if g_r_date else 'N/A'
            g_runtime = g_details.get('runtime', 0)
            g_overview = g_details.get('overview', 'No synopsis available.')
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(127,0,255,0.2); border-radius:16px; padding:20px; margin-bottom:25px;">
                <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                    <img src="{g_poster}" style="width:140px; border-radius:10px; box-shadow:0 8px 20px rgba(0,0,0,0.5);" />
                    <div style="flex:1; min-width:250px;">
                        <h2 style="margin:0; font-weight:700; color:#fff;">{g_title} ({g_year})</h2>
                        <p style="color:#00f0ff; font-style:italic; margin:4px 0 10px 0;">{g_tagline}</p>
                        <div style="margin-bottom:10px;">
                            <span style="color:#ffc107; font-weight:700; margin-right:15px;">⭐ {g_rating}/10</span>
                            <span style="color:#aaa;">⏱️ {g_runtime} min</span>
                        </div>
                        <p style="font-size:0.9rem; color:#ccc; line-height:1.5; margin:0;">{g_overview}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Analyzing graph nodes and computing similarity paths..."):
                global_recs = get_tmdb_recommendations(st.session_state.global_selected_movie_id)
                
            st.markdown("<h3 style='margin: 25px 0 15px 0; font-size: 1.4rem; font-weight: 700; color:#fff;'>🌐 Dynamic API Neural Mappings (TMDB Recommended)</h3>", unsafe_allow_html=True)
            
            filtered_global_items = []
            for item in global_recs:
                r_date = item.get('release_date', '')
                r_year = int(r_date.split('-')[0]) if (r_date and r_date.split('-')[0].isdigit()) else 0
                r_rating = item.get('vote_average', 0.0)
                
                if r_rating < sidebar_min_rating:
                    continue
                if r_year < sidebar_start_year or r_year > sidebar_end_year:
                    continue
                
                filtered_global_items.append(item)
            
            if not filtered_global_items:
                st.info("No neural matches found within your sidebar filter boundaries.")
            else:
                g_cols = st.columns(4)
                for g_idx, item in enumerate(filtered_global_items[:8]):
                    col_idx = g_idx % 4
                    with g_cols[col_idx]:
                        rec_id = item.get('id')
                        rec_title = item.get('title')
                        rec_poster = get_poster_url(item.get('poster_path'))
                        rec_date = item.get('release_date', '')
                        rec_year = rec_date.split('-')[0] if rec_date else 'N/A'
                        rec_rating = round(item.get('vote_average', 0.0), 1)
                        
                        card_html = f"""
                        <div class="movie-card" style="margin-bottom: 12px;">
                            <div class="movie-badge">⭐ {rec_rating}</div>
                            <img src="{rec_poster}" style="width: 100%; height: 260px; object-fit: cover; display: block;" />
                            <div style="padding: 10px;">
                                <h5 style="margin: 0; font-size: 0.9rem; color: #fff; font-weight: 700; min-height: 40px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">{rec_title} ({rec_year})</h5>
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                        if st.button("🎯 Select Node", key=f"sel_rec_gl_{rec_id}", use_container_width=True):
                            st.session_state.global_selected_movie_id = rec_id
                            st.rerun()

# ================= TAB 3: MOOD RECOMMENDER =================
with tabs[2]:
    st.markdown("<h3 style='margin: 0 0 10px 0; font-size: 1.4rem; font-weight: 700; color:#fff;'>🧠 Semantic Mood & Vibe Discovery</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.9rem; color:#aaa; margin-bottom:15px;'>Describe what you want to watch in natural language keywords (e.g., <i>'dark psychological serial killer twist ending'</i>).</p>", unsafe_allow_html=True)
    
    mood_input = st.text_input("Describe your vibe / keywords:", placeholder="e.g. mind-bending space magic marvel action", key="mood_search_input")
    
    if mood_input:
        mood_vector = cv.transform([mood_input]).toarray()
        mood_distances = cosine_similarity(mood_vector, vectors)[0]
        mood_results = sorted(list(enumerate(mood_distances)), reverse=True, key=lambda x: x[1])
        
        st.markdown("<h3 style='margin: 20px 0 15px 0; font-size: 1.25rem; font-weight: 700; color:#fff;'>🧠 Highly Matching Vibes</h3>", unsafe_allow_html=True)
        
        m_cols = st.columns(3)
        rendered_count = 0
        
        for idx, score in mood_results:
            if score < 0.05:
                continue
                
            m_id = df.iloc[idx]['movie_id']
            m_title = df.iloc[idx]['title']
            
            m_details = get_movie_details(m_id)
            if not m_details:
                continue
                
            r_date = m_details.get('release_date', '')
            r_year = int(r_date.split('-')[0]) if (r_date and r_date.split('-')[0].isdigit()) else 0
            r_rating = m_details.get('vote_average', 0.0)
            r_genres = [g['name'] for g in m_details.get('genres', [])]
            
            if r_rating < sidebar_min_rating:
                continue
            if r_year < sidebar_start_year or r_year > sidebar_end_year:
                continue
            if sidebar_genre != "All" and sidebar_genre not in r_genres:
                continue
                
            col_idx = rendered_count % 3
            with m_cols[col_idx]:
                genres_html = " ".join([f'<span class="genre-pill">{g}</span>' for g in r_genres[:2]])
                card_html = f"""
                <div class="movie-card" style="margin-bottom: 12px;">
                    <div class="movie-badge" style="color:#00f0ff;">✨ Match: {int(score*100)}%</div>
                    <img src="{get_poster_url(m_details.get('poster_path'))}" style="width: 100%; height: 300px; object-fit: cover; display: block;" />
                    <div style="padding: 15px;">
                        <h4 style="margin: 0 0 5px 0; font-size: 1.05rem; color: #fff; font-weight: 700; min-height: 48px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">{m_title}</h4>
                        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #aaa; margin-top: 5px; margin-bottom: 8px;">
                            <span>📅 {r_year}</span>
                            <span>⭐ {round(r_rating, 1)}</span>
                        </div>
                        <div style="min-height: 30px;">
                            {genres_html}
                        </div>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                if st.button("🔍 Explore Curated Vibe", key=f"mood_explore_{m_id}", use_container_width=True):
                    st.session_state.selected_movie_id = m_id
                    st.rerun()
                    
            rendered_count += 1
            if rendered_count >= 6:
                break