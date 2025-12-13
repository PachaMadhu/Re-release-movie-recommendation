"""
Movie Re-Release Prediction System - Streamlit App
Predicts if user will watch re-releases and recommends movies
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re # <-- NECESSARY IMPORT

# -------------------------------------------------------
# DATA CLEANING FUNCTIONS (ADDED FOR JOBLIB COMPATIBILITY)
# -------------------------------------------------------
def clean_text(text):
    """Clean text data and replace invalid entries with 'salaar'"""
    if pd.isna(text) or text is None:
        return 'salaar'
    
    text = str(text).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    
    invalid_entries = [
        'none', 'no', 'na', 'n/a', '.', '-', 'nothing', 
        'all', 'everyone', 'any', 'idk', 'don\'t know', 'not sure',
        'no favorite', 'no favourite', 'no one', 'all movies', 
        'all movies are my favourite', 'it\'s subjective', 'subjective',
        'tfi banisa', 'whloe tfi', 'whole tfi', 'all heros', 
        'i am all heros fan', 'no favourite i just watch everyone\'s movies',
        'based on movie', 'yes', 'nothing', 'all ', 'any pawan kalyan movie',
        'awaara, salaar and many more', 'saalar and bahubali', 
        'awaara, orange, salaar and many more', 'salaar,chakram'
    ]
    
    if text in invalid_entries or text == '' or len(text) <= 2:
        return 'salaar'
    
    if ',' in text or ' and ' in text or 'many more' in text:
        return 'salaar'
    
    return text

def standardize_actor_name(actor):
    """Standardize actor name variations"""
    if not actor or actor == 'salaar':
        return 'prabhas'
    
    actor = str(actor).strip().lower()
    
    actor_map = {
        'prabhas': ['prabhas', 'prabhass', 'prabas', 'rebel star prabhas','All','all','No favourite I just watch everyones movies','Whloe TFI','I am all heros fan','TFI banisa','No one ','No favorite','None','no one',''],
        'ram charan': ['ram charan', 'ramcharan', 'rc', 'cherry'],
        'jr. ntr': ['jr ntr', 'jr.ntr', 'ntr jr', 'tarak', 'young tiger', 'jr. ntr'],
        'allu arjun': ['allu arjun', 'alluarjun', 'aa', 'bunny', 'stylish star'],
        'mahesh babu': ['mahesh babu', 'maheshbabu', 'mahesh', 'superstar mahesh'],
        'pawan kalyan': ['pawan kalyan', 'pawankalyan', 'pk', 'power star'],
        'vijay deverakonda': ['vijay devarakonda', 'vijay deverakonda', 'vijay devarakonda ', 'vd'],
        'nani': ['nani', 'natural star nani'],
        'ravi teja': ['ravi teja', 'raviteja', 'mass maharaja'],
        'balakrishna': ['balaiah', 'balakrishna', 'nandamuri balakrishna', 'nbk'],
        'yash': ['yash', 'rocking star yash'],
        'bellamkonda srinivas': ['bellamkonda srinivas', 'bellamkonda'],
        'raghava lawrence': ['raghava lawrance', 'raghava lawrence', 'lawrence'],
        
        # Hindi actors
        'shah rukh khan': ['shah rukh khan', 'shahrukh khan', 'srk', 'king khan'],
        'salman khan': ['salman khan', 'salman', 'bhai'],
        'aamir khan': ['aamir khan', 'amir khan', 'aamir'],
        'hrithik roshan': ['hrithik roshan', 'hrithik', 'duggu'],
        'ranbir kapoor': ['ranbir kapoor', 'ranbir', 'rk'],
        'akshay kumar': ['akshay kumar', 'akshay', 'khiladi'],
        'sushant singh rajput': ['sushant singh rajput', 'sushant', 'ssr'],
        'varun dhawan': ['varun dhawan', 'varun dhawan ', 'varun'],
        'siddharth malhotra': ['siddharth malhotra', 'siddharth malhotra ', 'sidharth'],
        'irrfan khan': ['irfan khan', 'irrfan khan', 'irrfan'],
        
        # Tamil actors
        'vijay': ['vijay talapathy', 'vijay thalapathy', 'vijay', 'thalapathy'],
        
        # Other
        'r. madhavan': ['r. madhavan', 'madhavan', 'maddy'],
        'dulquer salmaan': ['dulquer salmaan', 'dulquer salmaan ', 'dq'],
        'lakshya': ['lakshya'],
        'kay kay menon': ['kk menon', 'kay kay menon'],
        'kareena kapoor': ['kareena kapoor', 'kareena kapoor ', 'bebo']
    }
    
    for standard_name, variations in actor_map.items():
        if actor in variations:
            return standard_name
    
    return actor

def standardize_movie_name(movie):
    """Standardize movie name variations"""
    if not movie:
        return 'salaar'
    
    movie = movie.strip().lower()
    movie = movie.replace('(telugu)', '').replace('()', '').strip()
    
    movie_map = {
        'baahubali': ['baahubali', 'bahubali', 'bahubali 1', 'bahubali 2', 'baahubali 1', 
                      'baahubali 2', 'bhahubali the epic', 'bahubali '],
        'rrr': ['rrr', 'r r r', 'rrrr','Rrr','RRR'],
        'pushpa': ['pushpa', 'pushpa 1', 'pushpa part 1'],
        'salaar': ['salaar', 'salar', 'salaar part 1', 'saalaar', 'salaar '],
        'kalki': ['kalki', 'kalki 2898 ad', 'kalki 2898', 'kalki, kgf'],
        'magadheera': ['magadheera', 'magadeera', 'magahdeera'],
        'rangasthalam': ['rangasthalam', 'rangastalam'],
        'pokiri': ['pokiri', 'pokkiri','Ssmb29'],
        'businessman': ['businessman', 'business man', 'buisnessman', 'businessman '],
        'khaleja': ['khaleja', 'khaleeja', 'khaleja '],
        'gabbar singh': ['gabbar singh', 'gabbar', 'gabbarsingh'],
        'attarintiki daredi': ['attarintiki daredi', 'attarintiki daaredi', 'ad'],
        'simhadri': ['simhadri', 'simahadri', 'shimhadri'],
        'temper': ['temper', 'temperr'],
        'saaho': ['saaho', 'sahooo'],
        'darling': ['darling', 'darling movie', 'darling '],
        'orange': ['orange', 'orange '],
        'jersey': ['jersey', 'jersey '],
        'color photo': ['color photo', 'color photo'],
        'arjun reddy': ['arjun reddy', 'arjun reddy '],
        'bhramotsavam': ['bhramotsavam', 'brahmotsavam'],
        'johnny': ['johnny', 'johnny '],
        'badri': ['badri', 'badri '],
        'chatrapati': ['chatrapati', 'chatrapati '],
        'happy days': ['hapoy days', 'happy days'],
        'sarrainodu': ['sarrinodu', 'sarrainodu', 'sarrinodu '],
        'oy': ['oy', 'oy!', 'oy! '],
        'one': ['one','1 nenokkadine'],
        'ala vaikunthapurramuloo': ['ala vaikuntapuram lo', 'ala vaikunthapurramuloo'],
        'raja varu rani garu': ['raja varu rani garu', 'raja varu rani garu '],
        'geetha govindam': ['gita govindam', 'geetha govindam'],
        'janatha garage': ['janatha garage'],
        'aravinda sametha': ['aravinda sametha', 'aravinda sametha '],
        'gang leader': ['gang leader', 'gang leader '],
        'panja': ['panja', 'panja '],
        'maryadha ramanna': ['maryadharaamanna'],
        '1 nenokkadine': ['1 nenokkadine'],
        '3 idiots': ['3 idiots', '3idiots', 'three idiots', '3 idiots '],
        'ddlj': ['ddlj', 'dilwale dulhania le jayenge'],
        'pathaan': ['pathaan', 'pathan'],
        'dil bechara': ['dil bechara', 'dil bechara', 'dilbechara', 'dil bechare', 
                         'dil bechara ', 'dil bechare '],
        'chhichhore': ['chhichhore', 'chichhore', 'chhichore', 'chhichhore '],
        'yeh jawaani hai deewani': ['yeh jwani hai deewani', 'ye jawaani hai dewaani', 
                                     'yeh jawaani hai deewani'],
        'kalank': ['kalank', 'kalank '],
        'sanam teri kasam': ['sanam teri kasam', 'sanam teri kasam-movie'],
        'spider-man': ['spiderman', 'spiderman ', 'spider-man no way home'],
        'avengers endgame': ['avengers:endgame', 'avengers endgame'],
        'avane srimannarayana': ['avane srimannarayana', 'avane srimannarayana'],
        'gaalipata': ['gaalipata', 'gaalipata '],
        'tumbbad': ['thumbad', 'tumbbad'],
        'hi nanna': ['hi nanna', 'hi nanna'],
        'adbhutham': ['adbutham', 'adbhutham'],
        'barfi': ['barfi', 'barfi!'],
        'style': ['style', 'style '],
        'jaya janaki nayaka': ['jaya janaki nayaka'],
        'lucky bhaskar': ['lucky bhaskar', 'lucky bhaskar '],
        'murari': ['murari', 'murari '],
        'inglourious basterds': ['inglorious bastard', 'inglourious basterds'],
        'kGF': ['k3g','KGF','Kgf','kgf','KGF 2','Kgf 2','kgf 2'],

    }
    
    for standard_name, variations in movie_map.items():
        if movie in variations:
            return standard_name
    
    return movie
# -------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------
st.set_page_config(
    page_title="Movie Re-Release Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 15px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
        transform: scale(1.05);
    }
    .movie-card {
        padding: 15px;
        border-radius: 8px;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin: 5px 0;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        model1 = joblib.load("logistic.joblib")
        model2 = joblib.load("Content_based_recomendation.joblib")
        label_encoders = joblib.load("label_encoders.joblib")
        scaler1 = joblib.load("scaler_interest.joblib")
        cleaning_funcs = joblib.load("cleaning_functions.joblib")
        return model1, model2, label_encoders, scaler1, cleaning_funcs
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.info("💡 Please make sure all model files are in the same directory:")
        st.code("""
- logistic.joblib
- Content_based_recomendation.joblib
- label_encoders.joblib
- scaler_interest.joblib
- cleaning_functions.joblib
""")
        return None, None, None, None, None

model1, model2, label_encoders, scaler1, cleaning_funcs = load_models()

# -------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------
def get_content_based_recommendations(user_fav_movie, user_fav_actor, user_language, top_n=5):
    """Get movie recommendations based on content similarity"""
    movie_database = model2['movie_database']
    cosine_sim = model2['cosine_sim']
    movie_to_idx = model2['movie_to_idx']
    idx_to_movie = model2['idx_to_movie']
    
    # Find the movie in database
    if user_fav_movie not in movie_to_idx:
        # Fallback: find movies with same actor
        similar_movies = movie_database[
            movie_database['fav_actor'].astype(str).str.contains(user_fav_actor, case=False, na=False)
        ]['Movie_choice'].tolist()
        
        if not similar_movies:
            # Fallback: same language
            similar_movies = movie_database[
                movie_database['Language'] == user_language
            ]['Movie_choice'].tolist()
        
        if similar_movies:
            # Use the first movie found as the anchor for recommendations
            user_fav_movie = similar_movies[0]
        else:
            # Ultimate Fallback: return the top movies from the dataset
            return [(movie, 0.5) for movie in movie_database['Movie_choice'].head(top_n).tolist()]
    
    movie_idx = movie_to_idx[user_fav_movie]
    sim_scores = list(enumerate(cosine_sim[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1] # [1:] excludes the movie itself
    
    movie_indices = [i[0] for i in sim_scores]
    recommendations = [
        (idx_to_movie[idx], sim_scores[i][1]) 
        for i, idx in enumerate(movie_indices)
    ]
    
    return recommendations

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.markdown("<h1>🎬 Movie Re-Release Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 1.2em;'>Discover if you'll watch a re-release and get personalized movie recommendations!</p>", unsafe_allow_html=True)
st.markdown("---")

# Check if models loaded
if model1 is None:
    st.stop()

# -------------------------------------------------------
# SIDEBAR - USER INPUT
# -------------------------------------------------------
st.sidebar.header("📝 Tell Us About Yourself")

# Age
age = st.sidebar.number_input(
    "Your Age",
    min_value=10,
    max_value=100,
    value=20,
    step=1
)

# Language/Industry
language = st.sidebar.selectbox(
    "Which film industry do you mostly follow? *",
    options=['Telugu', 'Hindi']
)

# Actor-Movie Database
actor_movies = {
    'Telugu': {
        'Ram Charan': ['Magadheera', 'RRR', 'Rangasthalam', 'Chirutha', 'Nayak'],
        'Jr. NTR': ['RRR', 'Janatha Garage', 'Temper', 'Aravinda Sametha', 'Simhadri'],
        'Allu Arjun': ['Pushpa', 'Ala Vaikunthapurramuloo', 'Sarrainodu', 'Arya', 'DJ: Duvvada Jagannadha'],
        'Prabhas': ['Baahubali', 'Mirchi', 'Saaho', 'Salaar', 'Kalki'],
        'Mahesh Babu': ['Pokiri', 'Srimanthudu', 'Businessman', 'Khaleja', 'Guntur Kaara'],
        'Pawan Kalyan': ['Gabbar Singh', 'Attarintiki Daredi', 'Jalsa', 'Kushi', 'OG']
    },
    'Hindi': {
        'Shah Rukh Khan': ['DDLJ', 'Chennai Express', 'Pathaan', 'Jawan', 'Chak De! India'],
        'Salman Khan': ['Bajrangi Bhaijaan', 'Sultan', 'Kick', 'Tiger Zinda Hai', 'Dabangg'],
        'Aamir Khan': ['3 Idiots', 'Dangal', 'PK', 'Lagaan', 'Ghajini'],
        'Hrithik Roshan': ['Krrish', 'War', 'Zindagi Na Milegi Dobara', 'Jodhaa Akbar', 'Super 30'],
        'Ranbir Kapoor': ['Yeh Jawaani Hai Deewani', 'Rockstar', 'Barfi!', 'Sanju', 'Animal'],
        'Akshay Kumar': ['Hera Pheri', 'Airlift', 'Kesari', 'Toilet: Ek Prem Katha', 'Housefull 4'],
        'Sushant Singh Rajput': ['Chhichhore', 'M.S Dhoni - The Untold Story', 'Raabta', 'Dil Bechara', 'Kai po che!']
    }
}

# Favorite Actor
actor_list = list(actor_movies[language].keys()) + ['Other']
fav_actor = st.sidebar.selectbox(
    "Favorite actor *",
    options=actor_list
)

# If Other is selected
if fav_actor == 'Other':
    fav_actor = st.sidebar.text_input(
        "Enter actor name:",
        value="",
        placeholder="Type actor name here..."
    )

# Favorite Movie
if fav_actor and fav_actor != 'Other' and fav_actor in actor_movies.get(language, {}):
    movie_list = actor_movies[language][fav_actor]
    fav_mov = st.sidebar.selectbox(
        "What is your favorite movie? *",
        options=movie_list
    )
else:
    fav_mov = st.sidebar.text_input(
        "What is your favorite movie? *",
        value="",
        placeholder="Enter your favorite movie"
    )

# Ticket Price
ticket_price = st.sidebar.selectbox(
    "Highest ticket price you paid (₹) *",
    options=['150', '300', '500', 'more than 1000']
)

# Watch Count
watch_count = st.sidebar.selectbox(
    "How many re-releases have you watched? *",
    options=['0', '3', 'More than 5']
)

# Interest/Motivation
interest = st.sidebar.selectbox(
    "What motivates you to watch a re-release? *",
    options=['Nostalgia', 'Actor', 'Music', 'Mass Scenes', 'Friends']
)

# Movie Preference
movie_pref = st.sidebar.selectbox(
    "Which do you enjoy more? *",
    options=['Normal movie releases', 'Re-releases', 'Both']
)

# Theatre Count
theatre_count = st.sidebar.selectbox(
    "How often do you go to theatres? *",
    options=['Rarely', 'Occasionally', 'Often']
)

# First Day First Show
fdfs = st.sidebar.radio(
    "Do you usually watch FDFS of re-releases? *",
    options=['Yes', 'No']
)

# Excitement Level
old_movies = st.sidebar.slider(
    "How excited are you when a blockbuster is re-released? (1-5) *",
    min_value=1,
    max_value=5,
    value=4,
    step=1
)

st.sidebar.markdown("---")

# Validation
can_predict = True
if not fav_actor or fav_actor.strip() == "":
    st.sidebar.warning("⚠️ Please enter your favorite actor")
    can_predict = False

if not fav_mov or fav_mov.strip() == "":
    st.sidebar.warning("⚠️ Please enter your favorite movie")
    can_predict = False

predict_button = st.sidebar.button("🎯 Predict Now!", use_container_width=True, disabled=not can_predict)

# -------------------------------------------------------
# PREDICTION LOGIC
# -------------------------------------------------------
if predict_button:
    
    # 1. Clean inputs using saved cleaning functions (RE-INSERTED LOGIC)
    clean_text = cleaning_funcs['clean_text']
    standardize_actor = cleaning_funcs['standardize_actor_name']
    standardize_movie = cleaning_funcs['standardize_movie_name']
    
    fav_actor_clean = standardize_actor(clean_text(fav_actor))
    fav_mov_clean = standardize_movie(clean_text(fav_mov))
    
    # 2. Create input data dictionary (RE-INSERTED LOGIC)
    input_data = {
        'Age': age, 
        'Ticket_price': ticket_price,
        'Watch_count': watch_count,
        'Old_movies': old_movies, 
        'Language': language,
        'Interest': interest,
        'Movie_pref': movie_pref,
        'Theatre_count': theatre_count,
        'fdfs': fdfs,
        'fav_mov': fav_mov_clean,
        'fav_actor': fav_actor_clean
    }
    
    # Prepare features for Model 1 (Classification)
    # This list has 10 features, matching the 10 features expected by the scaler.
    feature_order = [
        # Numerical feature (slider value)
        'Old_movies', 
        # Categorical features (should total 9)
        'Ticket_price', 
        'Watch_count', 
        'Language', 
        'Interest', 
        'Movie_pref',
        'Theatre_count', 
        'fdfs', 
        'fav_mov', 
        'fav_actor'
    ] # Total: 10 features
    
    X_input_list = []
    
    # Determine if the loaded object is a single encoder or a dictionary 
    is_single_encoder = not isinstance(label_encoders, dict)

    for feature in feature_order:
        # Use .get() on the newly defined input_data
        value = input_data.get(feature, None) 
        
        # Numerical features ('Old_movies' only now)
        if feature in ['Old_movies']:
            X_input_list.append(value)
        # Categorical features (to be encoded)
        else:
            try:
                # Fallback check for single LabelEncoder object
                if is_single_encoder:
                    le = label_encoders 
                else:
                    le = label_encoders[feature]
                
                # Transform the value
                encoded_val = le.transform([value])[0]
                X_input_list.append(encoded_val)
            except (KeyError, ValueError):
                # Handle unseen category or missing key by defaulting to 0
                X_input_list.append(0) 

    
    # Create feature array and scale
    X_input = np.array(X_input_list).reshape(1, -1)
    X_scaled = scaler1.transform(X_input) 
    
    # Model 1 Prediction
    interest_pred = model1.predict(X_scaled)[0]
    interest_proba = model1.predict_proba(X_scaled)[0]
    
    # Display Results
    st.markdown("## 🎯 Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎬 Will You Watch a Re-Release?")
        
        if interest_pred == 1:
            st.markdown(f"<h2 style='color: #4CAF50; text-align: center;'>✅ YES!</h2>", unsafe_allow_html=True)
            st.success(f"Confidence: {interest_proba[1]:.1%}")
        else:
            st.markdown(f"<h2 style='color: #f44336; text-align: center;'>❌ NO</h2>", unsafe_allow_html=True)
            st.error(f"Confidence: {interest_proba[0]:.1%}")
        
        st.progress(float(interest_proba[1]))
        st.caption(f"Yes: {interest_proba[1]:.1%} | No: {interest_proba[0]:.1%}")
    
    with col2:
        st.markdown("### 📊 Your Profile Summary")
        st.write(f"**Age:** {age} years")
        st.write(f"**Language:** {language}")
        st.write(f"**Max Ticket Price:** ₹{ticket_price}")
        st.write(f"**Re-releases Watched:** {watch_count}")
        st.write(f"**Excitement Level:** {old_movies}/5 ⭐")
        st.write(f"**Favorite Movie:** {fav_mov}")
        st.write(f"**Favorite Actor:** {fav_actor}")
    
    # Model 2: Movie Recommendations
    if interest_pred == 1:
        st.markdown("---")
        st.markdown("## 🎥 Recommended Movies for You")
        
        recommendations = get_content_based_recommendations(
            fav_mov_clean, 
            fav_actor_clean, 
            language, 
            top_n=5
        )
        
        # Display recommendations in cards
        cols = st.columns(5)
        
        for idx, (col, (movie, prob)) in enumerate(zip(cols, recommendations)):
            with col:
                st.markdown(f"""
                    <div class='movie-card'>
                        <h3>#{idx+1}</h3>
                        <h4>{movie.title()}</h4>
                        <p style='font-size: 1.2em;'>{prob:.1%}</p>
                        <p>Match Score</p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎬 Top Pick Details")
        st.info(f"**We highly recommend:** {recommendations[0][0].title()} (Match Score: {recommendations[0][1]:.1%})")
        st.write(f"Based on your preferences for **{fav_actor}** and **{fav_mov}**, this movie is perfect for you!")
    
    else:
        st.markdown("---")
        st.markdown("## 💡 Why You Might Not Be Interested")
        st.info("Based on your profile, you seem to prefer normal movie releases over re-releases. But you can always change your mind! 😊")

else:
    # Welcome Screen
    st.markdown("""
        <div style='background: white; padding: 30px; border-radius: 15px; margin: 20px 0;'>
            <h2 style='color: #667eea; text-align: center;'>Welcome! 👋</h2>
            <p style='text-align: center; font-size: 1.1em; color: #333;'>
                Fill in your movie preferences in the sidebar and click <strong>"Predict Now!"</strong> 
                to discover if you'll enjoy a movie re-release and get personalized recommendations!
            </p>
            <hr>
            <h3 style='color: #764ba2;'>How it works:</h3>
            <ol style='font-size: 1.1em; color: #333;'>
                <li><strong>Model 1</strong> (Classifier) predicts if you'll watch a re-release</li>
                <li><strong>Model 2</strong> (Content-Based Filtering) recommends movies using TF-IDF similarity</li>
                <li>Get personalized suggestions with confidence scores!</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    # Statistics
    st.markdown("## 📊 About Our Models")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="🎯 Model 1", value="Classifier", delta="Prediction of Interest")
    
    with col2:
        st.metric(label="🎬 Model 2", value="Content-Based", delta="TF-IDF + Cosine Similarity")
    
    with col3:
        st.metric(label="📊 Dataset", value="User Survey Data", delta="208 Responses")

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 20px;'>
        <p>Made with ❤️ using Streamlit | Powered by Machine Learning 🤖</p>
        <p style='font-size: 0.8em;'>© 2024 Movie Re-Release Predictor</p>
    </div>
""", unsafe_allow_html=True)