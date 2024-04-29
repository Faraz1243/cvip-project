from flask import Flask, request, jsonify
from app import model, predict_image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "images"

@app.route('/checkImg', methods=['POST'])
def validate():    
    if 'img' not in request.files:
        return jsonify({'error': 'No Image part'})
    file = request.files['img']
    # If the user submits an empty part without a file, ignore it
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Save the file to the uploads folder
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], "img.jpeg"))
    
    img = predict_image('./images/img.jpeg')
    
    if img:
        return jsonify({'result': 'The Brain Image contains Tumor'})
    else:
        return jsonify({'result': 'The Brain Image does not contains Tumor'})



if __name__ == '__main__':
    app.run(debug=True)