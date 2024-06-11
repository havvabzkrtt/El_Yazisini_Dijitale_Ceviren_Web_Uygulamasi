from flask import Flask, request, render_template, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
from src_code.main import process_image  
import cv2
import base64


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        surname = request.form['surname']
        form_type = request.form['form_type']
        session['name'] = name
        session['surname'] = surname
        session['form_type'] = form_type
        return redirect(url_for('upload'))
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'croppedImage' not in request.form:
            return "Cropped image not uploaded!"

        cropped_image_data = request.form['croppedImage']
        
        if not cropped_image_data:
            return "Cropped image data is missing!"

        cropped_image_data = cropped_image_data.split(',')[1]
        cropped_image_bytes = base64.b64decode(cropped_image_data)
        filename = secure_filename("cropped_image.png")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(file_path, "wb") as f:
            f.write(cropped_image_bytes)

        # Form bilgilerini session'dan alıyoruz
        name = session.get('name')
        surname = session.get('surname')
        form_type = session.get('form_type')
        
        if form_type == 'personal_info':
            selected_option_text = 'Personal Informations'
        elif form_type == 'uni_info':
            selected_option_text = 'University Informations'
        elif form_type == 'address_info':
            selected_option_text = 'Address Informations'

        # Resmi işleyip sonucu alıyoruz
        img = cv2.imread(file_path)
        result = process_image(form_type, img)
    
        print(type(result))
        print(result)
        result.update({"name": name, "surname": surname, "form_type": selected_option_text})
        return render_template('result.html', result=result)

    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)

