from flask import Flask, flash, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
import torch
import numpy as np
import matplotlib.pyplot as plt
 
app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5','custom',path='static/models/best.pt',force_reload=True,verbose=False)

 
UPLOAD_FOLDER = 'static/upload/'
 
app.secret_key = "itsmayurmore"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     


@app.route('/')
def home():
    return render_template('upload.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part' , category='warning')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading' , category='warning')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded and displayed below',category='info')
        global result
         
        result = model(UPLOAD_FOLDER+filename)
        plt.imshow(np.squeeze(result.render()))
        plt.savefig('static/saved/plot.png')
        
        return render_template('upload.html', filename='plot.png',output = result)
    else:
        flash('Allowed image types are - png, jpg, jpeg only' , category='warning')
        return redirect(request.url)

 
if __name__ == "__main__":
    app.run(debug=False)