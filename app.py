from flask import Flask, request, render_template, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
from src_code.main import process_image  # process_image fonksiyonunu içe aktarıyoruz
import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Form verilerini saklamak için gerekli
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
        if 'file' not in request.files:
            return "Dosya yüklenmedi"
        file = request.files['file']
        if file.filename == '':
            return "Dosya seçilmedi"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Form bilgilerini session'dan alıyoruz
            name = session.get('name')
            surname = session.get('surname')
            form_type = session.get('form_type')
            
            # Resmi işleyip sonucu alıyoruz
            img=cv2.imread(file_path)
            result = process_image(form_type, img)
            print(type(result))
            print(result)
            return render_template('result.html', result=result)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
