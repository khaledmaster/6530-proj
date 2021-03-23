from flask import Flask, render_template, request, redirect, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sql.db'
db = SQLAlchemy(app)

class audio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(200), unique=True, nullable=False)
    date_created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return '<Task %r>' % self.id

db.create_all()



@app.route('/', methods=['POST', 'GET'])
def index():
    # if request.method == 'POST':
    #     task_content = request.form['content']
    #     return redirect('/')
    #     # new_task = audio(content =task_content)
    #
    #     # try:
    #     #     db.session.add(new_task)
    #     #     db.session.commit()
    #     #     return redirect('/')
    #     # except:
    #     #     return 'There was an issue adding to the DB"'
    #
    # else:
    #     tasks = audio.query.order_by(audio.date_created).all()
        return render_template("audiolist.html")

@app.route('/check', methods=['POST', 'GET'])
def check():
    freqs = {
        'result': 'fake',
        'label1': 'fake',
        'label2': 'fake',
        'label3': 'fake',
        'label4': 'fake'
    }
    print(request.is_json)
    content = request.get_json()
    print(content)
    return jsonify(freqs)


if __name__ == "__main__":
    app.run(debug=True)