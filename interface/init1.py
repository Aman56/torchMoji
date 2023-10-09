from flask import Flask, render_template, request, session, url_for, redirect, jsonify, flash;
import pymysql.cursors;
import uuid;
import requests;
from emoji_MusicGen import gen_and_save;
import os;

app = Flask(__name__, static_folder = 'static', static_url_path = '/static')

# Configure the path where generated music files will be stored
app.config['MUSIC_FOLDER'] = os.path.join('static', 'pregenerated_music')
app.config['retrieve_MUSIC_FOLDER'] = 'pregenerated_music'

def get_connection():
    conn = pymysql.connect(host='192.168.64.2',
                       user='root',
                       password='',
                       db='emo_music',
                       charset='utf8mb4',
                       cursorclass=pymysql.cursors.DictCursor)
    return conn

def generate_user_id():
    userid = str(uuid.uuid4())
    username = 'user_' + userid
    return username

def insert_user_into_db(username):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                'INSERT INTO user (username) VALUES (%s)',
                (username,)
            )
            connection.commit()
    finally:
        connection.close()

def insert_comment_into_db(song_id,comment_post, username):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                'INSERT INTO comment (song_id,comment_post, username) VALUES (%s, %s, %s)',
                (song_id,comment_post, username)
            )
            connection.commit()
    finally:
        connection.close()

def get_comments_for_user(username, song_id):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute('SELECT comment_post FROM comment WHERE username = %s AND song_id = %s', (username,song_id))
            comments = [row['comment_post'] for row in cursor.fetchall()]
    finally:
        connection.close()
    return comments

@app.route('/')
def index():
    username = session.get('username')
    #print("username",username)
    
    song_id = request.args.get('song_id')
    if song_id is not None:
        session["song_id"] = song_id
    else:
        session["song_id"] = 0

    song_id = session["song_id"]
    if username:
        welcome_message = "Welcome, " + username
        comments = get_comments_for_user(username, song_id)
    else:
        welcome_message = "You haven't logged in"
        comments = []
    #print("songid", song_id,"comments",comments)
    return render_template('home.html', username = username, comments=comments, welcome_message = welcome_message)

@app.route('/get_username')
def get_username():
    username = session.get('username')
    return jsonify({'username': username})

@app.route('/submit_comment', methods=['POST'])
def submit_comment():
    comment = request.form.get('comment')
    song_id = request.form.get('song_id')
    username = session.get('username')
    
    if not username:
        username = generate_user_id()  # Generate a new username
        insert_user_into_db(username)  # Insert the new username into the database
        session['username'] = username
        
    insert_comment_into_db(song_id, comment, username)

    return jsonify({
        'song_id': song_id,
        'comment': comment,
        'username': username,
    })

@app.route('/generate_music', methods=['POST'])
def generate_music():
    user_input = request.json.get('user_input')
    emojis = [emoji for emoji in user_input if emoji.strip()]
    if user_input:
        file_path = gen_and_save(user_input)

    generated_music = {
        'name': "Generated Music",
        'path':file_path,
        'singer': None,
        'emojis':emojis
    }

    response = {
        "result": 'Music generated successfully',
        'music':generated_music,
    }

    #generated_music = response.json()["generated_music"]
    # Add the generated music to the playlist or return it as needed 
    #return jsonify({"generated_music": generated_music})
    return jsonify(response)

@app.route('/update_playlist', methods=['POST'])
def update_playlist():
    updated_playlist = request.json.get('playlist')

    # You can now save the updated_playlist to a file or database for persistence

    response = {
        "result": 'Playlist updated successfully',
        "updated_playlist": updated_playlist,
    }
    return jsonify(response)

@app.route('/static/pregenerated_music/')
def pregenerated_music_listing():
    try:
        music_folder = os.path.join(app.static_folder, 'pregenerated_music')
        files = os.listdir(music_folder)
        return jsonify(files)
    except Exception as e:
        print("An error occurred:", str(e))
        return jsonify(error=str(e)), 500 


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return jsonify({'message': 'Logged out'})
		
app.secret_key = 'some key that you will never guess'
app.config['MYSQL_CONNECT_TIMEOUT'] = 10
#Run the app on localhost port 5000
#debug = True -> you don't have to restart flask
#for changes to go through, TURN OFF FOR PRODUCTION
if __name__ == "__main__":
    #public_url = ngrok.connect(port=5000)
    app.run('127.0.0.1', 5000, debug = True)
