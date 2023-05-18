from flask import Flask, render_template
from data_utilis import server_base_path, dataset_root

app = Flask(__name__)


@app.route('/')
def start():  # put application's code here
    return render_template('result.html')


if __name__ == '__main__':
    app.run()
