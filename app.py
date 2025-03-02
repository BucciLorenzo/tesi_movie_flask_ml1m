from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import pickle as pkl
import pandas as pd 
import numpy as np 
import faiss
import uuid
from read_id_title_dataset_tsv import read_movie_tsv
import os
import logging
from flask import Flask, request, session, jsonify
from datetime import datetime
import time

ID_OFFSET = 6040  # Definizione della costante id primo item

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # For session encryption

# Configure Flask-Session to use server-side session storage
from flask_session import Session
app.config['SESSION_TYPE'] = 'filesystem'  # Use filesystem for storing session data
Session(app)

@app.route('/survey')
def survey():
    return render_template('survey.html')

def bprint_yellow(text : str):
    print(f"\033[0;30;43m{text}\033[0m")

# load dictionary ID-title
file_path = 'data/dataset.tsv'
id_to_title = read_movie_tsv(file_path)

# load data
dataset = pd.read_csv('data/dataset.tsv', sep='\t')
graph_embeddings = pkl.load(open('data/graph_embs.pkl', 'rb'))
sbert_embeddings = pkl.load(open('data/graph_ui_embs.pkl', 'rb'))
compgcn_embeddings = pkl.load(open('data/compgcn_384_n1.pkl','rb'))
compgcn_embeddings = { k: compgcn_embeddings[k] for k in compgcn_embeddings.keys() if k >= 6040 } # load item 
compgcn_embeddings = {id_to_title.get(str(k), k): v for k, v in compgcn_embeddings.items()}

vit_cls_embeddings = pkl.load(open('data/vit_cls.pkl','rb'))
vit_cls_embeddings = { k: vit_cls_embeddings[k] for k in vit_cls_embeddings.keys() if k >= 6040 }
vit_cls_embeddings = {id_to_title.get(str(k), k): v for k, v in vit_cls_embeddings.items()}

vggish_embeddings = pkl.load(open('data/vggish.pkl', 'rb'))
vggish_embeddings = {int(k): v for k, v in vggish_embeddings.items() if int(k) >= 6040}
vggish_embeddings = {id_to_title.get(str(k), k): v for k, v in vggish_embeddings.items()}

r2p1d_embeddings = pkl.load(open('data/r2p1d.pkl','rb'))
r2p1d_embeddings = {int(k): v for k, v in r2p1d_embeddings.items() if int(k) >= 6040}
r2p1d_embeddings = {id_to_title.get(str(k), k): v for k, v in r2p1d_embeddings.items()}

mini_embeddings = pkl.load(open('data/all-MiniLM-L12-v2_dictionary.pkl','rb'))
mini_embeddings = {int(k): v for k, v in mini_embeddings.items() if int(k) >= 6040}
mini_embeddings = {id_to_title.get(str(k), k): v for k, v in mini_embeddings.items()}

# efficient data structures for graph embeddings
graph_keys = list(graph_embeddings.keys())
graph_embedding_matrix = np.array(list(graph_embeddings.values()))
graph_embedding_dimension = graph_embedding_matrix.shape[1]
graph_index = faiss.IndexFlatL2(graph_embedding_dimension)
faiss.normalize_L2(graph_embedding_matrix) 
graph_index.add(graph_embedding_matrix)

# efficient data structures for text embeddings
sbert_keys = list(sbert_embeddings.keys())
sbert_embedding_matrix = np.array(list(sbert_embeddings.values()))
sbert_embedding_dimension = sbert_embedding_matrix.shape[1]
sbert_index = faiss.IndexFlatL2(sbert_embedding_dimension)
faiss.normalize_L2(sbert_embedding_matrix) 
sbert_index.add(sbert_embedding_matrix)

# efficient data structures for graph embeddings
compgcn_keys = list(compgcn_embeddings.keys())
compgcn_embedding_matrix = np.array(list(compgcn_embeddings.values()))
compgcn_embedding_matrix = compgcn_embedding_matrix.astype(np.float32)
compgcn_embedding_dimension = compgcn_embedding_matrix.shape[1]
compgcn_index = faiss.IndexFlatL2(compgcn_embedding_dimension)
faiss.normalize_L2(compgcn_embedding_matrix)
compgcn_index.add(compgcn_embedding_matrix)

# efficient data structures for images embeddings
vit_cls_keys = list(vit_cls_embeddings.keys())
vit_cls_embedding_matrix = np.array(list(vit_cls_embeddings.values()))
vit_cls_embedding_matrix = vit_cls_embedding_matrix.astype(np.float32)
vit_cls_embedding_dimension = vit_cls_embedding_matrix.shape[1]
vit_cls_index = faiss.IndexFlatL2(vit_cls_embedding_dimension)
faiss.normalize_L2(vit_cls_embedding_matrix)
vit_cls_index.add(vit_cls_embedding_matrix)

# efficient data structures for audio embeddings
vggish_keys = list(vggish_embeddings.keys())
vggish_embedding_matrix = np.array(list(vggish_embeddings.values()))
vggish_embedding_matrix = vggish_embedding_matrix.astype(np.float32)
vggish_embedding_dimension = vggish_embedding_matrix.shape[1]
vggish_index = faiss.IndexFlatL2(vggish_embedding_dimension)
faiss.normalize_L2(vggish_embedding_matrix)
vggish_index.add(vggish_embedding_matrix)

# Efficient data structures for video embeddings
r2p1d_keys = list(r2p1d_embeddings.keys())
r2p1d_embedding_matrix = np.array(list(r2p1d_embeddings.values()))
r2p1d_embedding_matrix = r2p1d_embedding_matrix.astype(np.float32)
r2p1d_embedding_dimension = r2p1d_embedding_matrix.shape[1]
r2p1d_index = faiss.IndexFlatL2(r2p1d_embedding_dimension)
faiss.normalize_L2(r2p1d_embedding_matrix)
r2p1d_index.add(r2p1d_embedding_matrix)

# Efficient data structures for text embeddings
mini_keys = list(mini_embeddings.keys())
mini_embedding_matrix = np.array(list(mini_embeddings.values()))
mini_embedding_matrix = mini_embedding_matrix.astype(np.float32)
mini_embedding_dimension = mini_embedding_matrix.shape[1]
mini_index = faiss.IndexFlatL2(mini_embedding_dimension)
faiss.normalize_L2(mini_embedding_matrix)
mini_index.add(mini_embedding_matrix)

# build movie catalog - Es 0 Jumanji
movie_catalog = []
for i, row in dataset.iterrows():
    movie_catalog.append({'id': i, 'title': row['Title']})
movie_titles_dataset = list(set(dataset['Title']))
movie_titles = list(set(dataset['Title']))


# build movie catalog with id_items title items - Es 6040 Jumanji
# serve a mappare l'id originale dell'item nel file di log
movie_catalog_id_title = []
for i, row in dataset.iterrows():
    movie_catalog_id_title.append({'id': row['id'], 'title': row['Title']})
movie_titles_dataset = list(set(dataset['Title']))
movie_titles = list(set(dataset['Title']))

# build profile
user_profile = {
    'username': 'User',
    'movies': [],
    'graph_embeddings': [],
    'sbert_embeddings': [],
    'compgcn_embeddings': [],
    'vit_cls_embeddings' : [],
    'vggish_embeddings' : [],
    'r2p1d_embeddings' : [],
    'mini_embeddings' : []
}


# vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movie_titles)

# search most similar movies by title
def vsm_query(query, limit=20):

    bprint_yellow(f'query {query}')

    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    fuzzy_scores = [fuzz.ratio(query.lower(), title.lower()) for title in movie_titles]
    combined_scores = 0.7 * cosine_similarities + 0.3 * (np.array(fuzzy_scores) / 100)
    ranked_indices = np.argsort(combined_scores)[::-1]

    results = []
    for id in ranked_indices[0:limit]:
        row = dataset[dataset['Title'] == movie_titles[id]]
        results.append({'id': str(row.index[0]),                #results.append({'id': str(row.index[0]), 
                        'title': row['Title'].values[0]})

    return results


# get recommendation
def get_recommendation(graph_user_emb, sbert_user_emb, compgcn_user_emb, vit_cls_user_emb, vggish_user_emb, r2p1d_user_emb, mini_user_emb, user_liked_movies, top_k=5):

    # convert embedding in float32
    graph_user_emb = graph_user_emb.astype(np.float32)
    sbert_user_emb = sbert_user_emb.astype(np.float32)
    compgcn_user_emb = compgcn_user_emb.astype(np.float32)
    vit_cls_user_emb = vit_cls_user_emb.astype(np.float32)
    vggish_user_emb = vggish_user_emb.astype(np.float32)
    r2p1d_user_emb = r2p1d_user_emb.astype(np.float32)
    mini_user_emb = mini_user_emb.astype(np.float32)

    # user embedding
    faiss.normalize_L2(graph_user_emb.reshape(1, -1))
    faiss.normalize_L2(compgcn_user_emb.reshape(1, -1))
    faiss.normalize_L2(vit_cls_user_emb.reshape(1, -1))
    faiss.normalize_L2(vggish_user_emb.reshape(1, -1))
    faiss.normalize_L2(r2p1d_user_emb.reshape(1, -1))
    faiss.normalize_L2(mini_user_emb.reshape(1, -1))

    # Search for the top_k most similar graph and text embeddings
    _, graph_indices = graph_index.search(graph_user_emb.reshape(1, -1), top_k+len(user_liked_movies))

    # drop the already liked movies
    filtered_graph_keys = []
    for idx in graph_indices[0]:
        key = graph_keys[idx]
        if key not in user_liked_movies:
            filtered_graph_keys.append(key)
        if len(filtered_graph_keys) == top_k:
            break

    _, sbert_indices = sbert_index.search(sbert_user_emb.reshape(1, -1), top_k+len(user_liked_movies))

    # drop the already liked movies
    filtered_sbert_keys = []
    for idx in sbert_indices[0]:
        key = sbert_keys[idx]
        if key not in user_liked_movies:
            filtered_sbert_keys.append(key)
        if len(filtered_sbert_keys) == top_k:
            break

    _, compgcn_indices = compgcn_index.search(compgcn_user_emb.reshape(1, -1), top_k+len(user_liked_movies))

    # drop the already liked movies
    filtered_compgcn_keys = []
    for idx in compgcn_indices[0]:
        key = compgcn_keys[idx]
        if key not in user_liked_movies:
            filtered_compgcn_keys.append(key)
        if len(filtered_compgcn_keys) == top_k:
            break

    _, vit_cls_indices = vit_cls_index.search(vit_cls_user_emb.reshape(1, -1), top_k+len(user_liked_movies))

    # drop the already liked movies
    filtered_vit_cls_keys = []
    for idx in vit_cls_indices[0]:
        key = vit_cls_keys[idx]
        if key not in user_liked_movies:
            filtered_vit_cls_keys.append(key)
        if len(filtered_vit_cls_keys) == top_k:
            break

    _, vggish_indices = vggish_index.search(vggish_user_emb.reshape(1, -1), top_k+len(user_liked_movies))

    # drop the already liked movies
    filtered_vggish_keys = []
    for idx in vggish_indices[0]:
        key = vggish_keys[idx]
        if key not in user_liked_movies:
            filtered_vggish_keys.append(key)
        if len(filtered_vggish_keys) == top_k:
            break

    _, r2p1d_indices = r2p1d_index.search(r2p1d_user_emb.reshape(1, -1), top_k+len(user_liked_movies))

    # drop the already liked movies
    filtered_r2p1d_keys = []
    for idx in r2p1d_indices[0]:
        key = r2p1d_keys[idx]
        if key not in user_liked_movies:
            filtered_r2p1d_keys.append(key)
        if len(filtered_r2p1d_keys) == top_k:
            break

    _, mini_indices = mini_index.search(mini_user_emb.reshape(1, -1), top_k + len(user_liked_movies))

    # drop the already liked movies
    filtered_mini_keys = []
    for idx in mini_indices[0]:
        key = mini_keys[idx]
        if key not in user_liked_movies:
            filtered_mini_keys.append(key)
        if len(filtered_mini_keys) == top_k:
            break


    return filtered_graph_keys, filtered_sbert_keys, filtered_compgcn_keys, filtered_vit_cls_keys, filtered_vggish_keys, filtered_r2p1d_keys, filtered_mini_keys

# Initialize session with user_id if it doesn't exist
def initialize_user_session(): 
    if 'user_id' not in session: 
        session['user_id'] = str(uuid.uuid4()) 
        session['user_profile'] = { 
            'username': 'User', 
            'movies': [], 
            'graph_embeddings': [], 
            'sbert_embeddings': [], 
            'compgcn_embeddings': [], 
            'vit_cls_embeddings': [], 
            'vggish_embeddings': [], 
            'r2p1d_embeddings': [], 
            'mini_embeddings': [], 
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        }

@app.before_request 
def before_request(): 
    initialize_user_session()

# Home route: Displays search form
@app.route('/')
def home():
    return render_template('home.html')

# Search route: Handles movie searches
@app.route('/search', methods=['POST'])
def search_movies():
    search_term = request.form.get('search_term')
    results = vsm_query(search_term)
    bprint_yellow(f'results: {results}')
    return render_template('search_results.html', movies=results, search_term=search_term)

@app.route('/live_search')
def live_search():
    query = request.args.get('query', '')
    
    if query:
        results = vsm_query(query)  # Use the existing search function
        bprint_yellow(f'results: {results}')
        return jsonify({'movies': results})
    else:
        return jsonify({'movies': []})


@app.route('/add_movie/<int:movie_id>', methods=['POST'])
def add_movie(movie_id):
    movie = next((m for m in movie_catalog if m['id'] == movie_id), None)
    
    if movie and movie not in session['user_profile']['movies']:
        session['user_profile']['movies'].append(movie)
        bprint_yellow(f"FILM AGGIUNTO: {movie}")
        session['user_profile']['graph_embeddings'].append(graph_embeddings[movie['title']])
        session['user_profile']['sbert_embeddings'].append(sbert_embeddings[movie['title']])
        session['user_profile']['compgcn_embeddings'].append(compgcn_embeddings[movie['title']])
        session['user_profile']['vit_cls_embeddings'].append(vit_cls_embeddings[movie['title']])
        session['user_profile']['vggish_embeddings'].append(vggish_embeddings[movie['title']])
        session['user_profile']['r2p1d_embeddings'].append(r2p1d_embeddings[movie['title']])
        session['user_profile']['mini_embeddings'].append(mini_embeddings[movie['title']]) 

        #Logga l'aggiunta di un film
        logging.info(f"Film aggiunto: {movie['title']} (ID: {movie['id']})")
    
    # Return JSON response instead of redirecting
    return jsonify({'status': 'success', 'movie': movie})

# View user's profile with saved movies
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    return render_template('profile.html', profile=session['user_profile'])


@app.route('/recommendation', methods=['POST'])
def recommendation(): 
    if session['user_profile']['movies']:

        top_k = 5
        # Costruire l'embedding del profilo utente 
        graph_user_emb = np.mean(session['user_profile']['graph_embeddings'], axis=0) 
        liked_titles = [movie['title'] for movie in session['user_profile']['movies']]

        sbert_user_emb = np.mean(session['user_profile']['sbert_embeddings'], axis=0) 
        compgcn_user_emb = np.mean(session['user_profile']['compgcn_embeddings'], axis=0) 
        vit_cls_user_emb = np.mean(session['user_profile']['vit_cls_embeddings'], axis=0) 
        vggish_user_emb = np.mean(session['user_profile']['vggish_embeddings'], axis=0) 
        r2p1d_user_emb = np.mean(session['user_profile']['r2p1d_embeddings'], axis=0) 
        mini_user_emb = np.mean(session['user_profile']['mini_embeddings'], axis=0)

        rec_graph, rec_sbert, rec_compgcn, rec_vit_cls, rec_vggish, rec_r2p1d, rec_mini = get_recommendation(
            graph_user_emb, sbert_user_emb, compgcn_user_emb, vit_cls_user_emb, vggish_user_emb, 
            r2p1d_user_emb, mini_user_emb, liked_titles, top_k=top_k
        )

        # Salva gli ID delle raccomandazioni nella sessione
        session['graph_ids'] = [str(dataset[dataset['Title'] == key].iloc[0]['id']) for key in rec_graph] 
        session['sbert_ids'] = [str(dataset[dataset['Title'] == key].iloc[0]['id']) for key in rec_sbert]
        session['compgcn_ids'] = [str(dataset[dataset['Title'] == key].iloc[0]['id']) for key in rec_compgcn]
        session['vit_ids'] = [str(dataset[dataset['Title'] == key].iloc[0]['id']) for key in rec_vit_cls]
        session['vggish_ids'] = [str(dataset[dataset['Title'] == key].iloc[0]['id']) for key in rec_vggish]
        session['r2p1d_ids'] = [str(dataset[dataset['Title'] == key].iloc[0]['id']) for key in rec_r2p1d]
        session['minilm_ids'] = [str(dataset[dataset['Title'] == key].iloc[0]['id']) for key in rec_mini]

        # Log per verificare le liste
        bprint_yellow(f"Graph Titles: {session['graph_ids']}")

        return jsonify({'status': 'success', 
                        'graph_recommendations': [{'title': dataset[dataset['Title'] == key].iloc[0]['Title']} for key in rec_graph], 
                        'sbert_recommendations': [{'title': dataset[dataset['Title'] == key].iloc[0]['Title']} for key in rec_sbert], 
                        'compgcn_recommendations': [{'title': dataset[dataset['Title'] == key].iloc[0]['Title']} for key in rec_compgcn], 
                        'vit_cls_recommendations': [{'title': dataset[dataset['Title'] == key].iloc[0]['Title']} for key in rec_vit_cls], 
                        'vggish_recommendations': [{'title': dataset[dataset['Title'] == key].iloc[0]['Title']} for key in rec_vggish], 
                        'r2p1d_recommendations': [{'title': dataset[dataset['Title'] == key].iloc[0]['Title']} for key in rec_r2p1d], 
                        'mini_recommendations': [{'title': dataset[dataset['Title'] == key].iloc[0]['Title']} for key in rec_mini]})
    else: 
        return jsonify({'status': 'success', 
                        'graph_recommendations': [], 
                        'sbert_recommendations': [], 
                        'compgcn_recommendations': [], 
                        'vit_cls_recommendations': [], 
                        'vggish_recommendations': [], 
                        'r2p1d_recommendations': [], 
                        'mini_recommendations': []})



@app.route('/clear', methods=['POST'])
def clear(): 
    session['user_profile'] = { 
        'username': 'User', 
        'movies': [], 
        'graph_embeddings': [], 
        'sbert_embeddings': [], 
        'compgcn_embeddings': [], 
        'vit_cls_embeddings': [], 
        'vggish_embeddings': [], 
        'r2p1d_embeddings': [], 
        'mini_embeddings': [], 
        'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S') 
    } 
    return jsonify({'status': 'success'})

@app.route('/profile_content')
def profile_content():
    return render_template('profile_content.html', profile=session['user_profile'])

################################## DEFINIZIONE DEL LOG ###################################################
# Directory per i file di log
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)  # Crea la directory

class CustomFileHandler(logging.FileHandler):
    def emit(self, record):
        # Evita l'aggiunta di \n alla fine del messaggio
        msg = self.format(record)
        self.stream.write(msg)  # Scrive il messaggio senza \n finale
        self.flush()  # Forza la scrittura immediata

def setup_logging(session_id):  
    log_filename = os.path.join(LOG_DIR, f"{session_id}.log")
    logger = logging.getLogger(session_id)
    
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        
        # Crea il CustomFileHandler
        file_handler = CustomFileHandler(log_filename)
        
        # Definisce il formato del log (senza newline finale)
        formatter = logging.Formatter('%(message)s', datefmt='%Y-%m-%d %H:%M:%S,%f')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
    return logger


# Inizializza il logger per ogni sessione
@app.before_request
def init_session_logger():
    if 'session_id' not in session:
        session['session_id'] = datetime.now().strftime('%Y%m%d%H%M%S%f')
    session['logger'] = setup_logging(session['session_id'])

###########################################################################################################

# Lista per memorizzare le risposte
responses = []

@app.route('/')
def index():
    return render_template('survey.html')

#Endpoint per i dati del sondaggio
@app.route('/submit', methods=['POST'])
def submit():
    user_responses = []

    # Raccogli le risposte
    for i in range(1, 15):  # 14 domande
        answer = request.form.get(f'q{i}')
        if answer:
            user_responses.append(int(answer))

    # Stampa in giallo nel terminale
    print("\033[93m", user_responses, "\033[0m")  

    # Salva le risposte nella sessione
    session['user_responses'] = user_responses

    # Recupera le risposte salvate in sessione
    user_responses = session.get('user_responses', [])

    # Ottieni il logger per la sessione corrente
    session_id = session.get('session_id')
    logger = setup_logging(session_id)

    timestamp = int(time.time() * 1000)  # Millisecondi
    log_message = "###" + ";;;".join(map(str, user_responses))

    logger.info(log_message)

    #Va a capo dopo aver scritto le risposte dell'utente**
    logger.info("\n")  # Aggiungi un ritorno a capo qui, dopo aver scritto le risposte

    # Messaggio di feedback per l'utente
    return '''
    <script>
        alert("Thank you for completing the survey! The page will close in 2 seconds.");
        setTimeout(function() { window.close(); }, 2000);
    </script>
    '''

# Endpoint per il pulsante "Valuta"
@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    algorithm_name = data.get('algorithm_name')
    timestamp = int(time.time() * 1000)  # Millisecondi

    # Ottiene i film degli utenti dalla sessione
    user_movies = session['user_profile'].get('movies', [])

    # Ottengo gli id dei film del profilo utente da stampare sul log
    # Ottengo gli id originali dei film mappando gli id del catalogo che parte da 0 con il catalogo con i veri id
    movie_ids = [
        catalog['id'] for catalog in movie_catalog_id_title
        if any(catalog['title'] == movie['title'] for movie in user_movies)
    ]

    # Ottieni gli ID delle raccomandazioni dalla sessione
    titles_map = {
        'Graph Recommendations': session.get('graph_ids', []),
        'Sbert Recommendations': session.get('sbert_ids', []),
        'Compgcn Recommendations': session.get('compgcn_ids', []),
        'Vit cls Recommendations': session.get('vit_ids', []),
        'Vggish Recommendations': session.get('vggish_ids', []),
        'R2p1d Recommendations': session.get('r2p1d_ids', []),
        'Mini Recommendations': session.get('minilm_ids', []),
    }

    # Inizializza la variabile per i titoli generici
    selected_titles = []

    # Assegna i titoli in base al nome dell'algoritmo
    if algorithm_name in titles_map:
        selected_titles = titles_map[algorithm_name]

    # Logga le informazioni
    logger = session['logger']
    logger.info(f"{timestamp}###{algorithm_name}###{';;;'.join(map(str, movie_ids))}###{';;;'.join(selected_titles)}")

    return jsonify({'status': 'success', 'message': 'Logged evaluation!'})

if __name__ == '__main__':
    app.run(debug=True)