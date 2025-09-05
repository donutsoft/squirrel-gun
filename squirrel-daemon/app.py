from flask import Flask, send_file, abort

app = Flask(__name__, static_url_path='')

@app.route('/')
def index():
    return "hello world"
