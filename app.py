import base64
import os
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from fastai.vision.all import *

# Output directory to save the uploaded images
OUTPUT_DIR = os.path.join(os.getcwd(), "output")

# Basic stylesheet, more imports at the style.css
external_stylesheets = [
	{
		"href": "https://codepen.io/chriddyp/pen/bWLwgP.css",
		"rel": "stylesheet",
	},
]

# Load the model
learn_inf = load_learner(Path()/"models/densenet121_pet_class.pkl", cpu=True)
# Seperate cat breeds from dog breeds
breeds = learn_inf.dls.vocab
cats = [breed.replace('_', ' ')
		for breed in breeds if breed[0].isupper()]
dogs = [breed.replace('_', ' ').title() 
		for breed in breeds if breed[0].islower()]

# Crate the web app and all its HTML layers
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Various cat and dog breeds recognition!"
server = app.server
app.layout = html.Div(
	    children=[
		html.Div(
			children=[
	        html.H1(children="Cat & Dog Breed Recognizer",
					className="header-title"),
	        html.P(
	            children=["Let me find the breed of your cat or dog.", 
						  " I can recognize various different cat and dog breeds.",
						  " I am pretty accurate about my recognition capabilities.",
						  " I will even provide a percentage of my certainty.",
						  html.Br(),
						  html.Small("If you hover over the result panel, a list of all the recognizable breeds are shown in a tooltip."),
						],
	            className="header-description"
	        ),
	        ],
	        className="header",
	        ),
	        html.Div(
				dcc.Upload(
				    id='upload-image',
					children=html.Div([
				    'Drag and Drop or ',
				    html.A(['Select an image of a pet'], style={"color": "hsl(23, 48%, 85%)"}),
					]),
				className="upload-button",
				multiple=False
	    		),
				title="Upload an image",
				className="center",	
	    	),
	    	html.Div(id="output-image-upload"),
	    	html.Div(["Created by theros88, based on the material provided at the fastai course ",
	    		html.A('"Deep Learning for Coders (2020)".', href='https://course.fast.ai/')],
	    		className="footer")
		])

def parse_contents(contents, filename, dir):
	"""Loads the filename in dir, makes the prediction and presents 
	   the image and the results in a panel."""
	img = PILImage.create(os.path.join(dir, filename))
	pred, pred_idx, probs = learn_inf.predict(img)
	# Output string formulation
	animal = "cat" if str(pred)[0].isupper() else "dog"
	s_pred = str(pred).replace('_', ' ').title()
	pred_str = f'{s_pred} ({probs[pred_idx]*100.:.01f}%)'

	# Returns the HTML image and the results panel
	return html.Div(
		children = [
			# HTML images accept base64 encoded strings in the same format
			# that is supplied by the upload
			html.Img(src=contents, title=filename, className="uploaded_img"),
			html.Div(children=[
					html.P(f"I reckon it's a {animal}. Its breed is:", style={"fontSize" : "18px"}),
					html.P(pred_str), 
					],
					className="w3-panel w3-round-xlarge prediction-description", 
					title='Prediction over the following breads:\n' + \
					       f"Cats: {', '.join(cats)}.\n\n" + f"Dogs: {', '.join(dogs)}." 
			)],
			className="center",	
		)


def save_file(name, content):
	"""Decode and store a file uploaded with Plotly Dash."""
	data = content.encode("utf8").split(b";base64,")[1]
	with open(name, "wb") as fp:
		fp.write(base64.decodebytes(data))

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'))
def update_output(content, name):
	if content is not None:
		save_file(os.path.join(OUTPUT_DIR, name), content)
		return parse_contents(content, name, OUTPUT_DIR)


if __name__ == '__main__':
	# Create the output directory if not present
	if not os.path.isdir(OUTPUT_DIR): 
		try: 
			os.mkdir(OUTPUT_DIR) 
		except OSError as error: 
			print(error)

	app.run_server(debug=True)

