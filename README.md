Movie Re-Release Prediction & Recommendation System With the growing trend of movie re-releases in India
this project aims to answer two practical questions: 

Will a user watch a movie re-release?
Which movie should be recommended to the user? 
The system is built using real survey data and follows a two-model machine learning approach.

📌 Project Overview 
This project uses 208 real user responses collected via Google Forms to build:

🔹 Model 1:

Interest Prediction (Classification) Algorithm:Logistic Regression 
Accuracy: ~78%
Goal: Predict whether a user is interested in watching a re-release 

Key Features: 
Age, Ticket, price, Theatre visit frequency, Favourite actor, Interest type (nostalgia, actor, story, etc.)

🔹 Model 2: 

Movie Recommendation System Approach:
Content-Based Filtering Techniques: TF-IDF Vectorization + Cosine Similarity 
Goal: Recommend Top 5 movies based on user preferences 
Uses: Favourite actors Favourite movies Language User interests 
 
✔️ Works well even with limited data (only ~80 movies)
✔️ Avoids the cold-start problem common in collaborative filtering

🛠 Tech Stack
Python 
Pandas & NumPy 
Scikit-learn 
TF-IDF Vectorizer 
Cosine Similarity 
Streamlit (for UI)

📊 Dataset Source:
Google Forms survey Responses: 208 users Survey

Link: 👉 https://lnkd.in/gkrNNqqg 
  

Check this Form if you are intrested
