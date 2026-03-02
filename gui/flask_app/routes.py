from flask import render_template
from gui.flask_app import app

@app.route('/')
@app.route('/index')
def index():
	machine = {
		'name': 'versal-1',
		'benchmark': 'efficient-det',
	}

	return render_template(
		'index.html',
		title = 'Home',
		machine = machine,
	)