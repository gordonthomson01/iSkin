import os
import io
import numpy as np
import random
import base64

from os.path import abspath

import keras
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications.xception import (
    Xception, preprocess_input, decode_predictions)
from keras.applications.vgg19 import(
    VGG19,
    preprocess_input,
    decode_predictions
)

from keras import backend as K

from flask import Flask, request, redirect, url_for, jsonify, render_template

app = Flask(__name__,static_url_path = "/static", static_folder = "static")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['IMAGE_FOLDER']='Images'
app.config['STATIC_FOLDER']='static'

model = None
graph = None

diag_list=['Melonoma','Bassal','Squamous','nevus','Unknown']
diag_type=['Benign','Malignant','Unknown']


def load_model():
    global model
    global graph
    #model = keras.models.load_model('mymodel-2.h5')
    #model = keras.models.load_model('Model1-CNNScratch.mlmodel')
    model = VGG19(include_top=True, weights='imagenet')
    graph = K.get_session().graph


load_model()


def prepare_image(img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # return the processed image
    return img


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    data = {"success": False}
    if request.method == 'POST':
        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename
            print(filename)

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #file.save(filepath)

            #move to static folder
            filepath=os.path.join(app.config['STATIC_FOLDER'], filename)
            file.save(filepath)

            # Load the saved image using Keras and resize it to the Xception
            # format of 299x299 pixels
            image_size = (224, 224)
            im = keras.preprocessing.image.load_img(filepath,
                                                    target_size=image_size,
                                                    grayscale=False)

            # preprocess the image and prepare it for classification
            image = prepare_image(im)

            global graph
            with graph.as_default():
                preds = model.predict(image)
                results = decode_predictions(preds)
                # print the results
                print(results)

                data['img_src']= filepath

                ### Rendering Plot in Html
                #figdata_png = base64.b64encode(file.getvalue())
                #data['img_html'] = figdata_png


                data["predictions"] = []

                # loop over the results and add them to the list of
                # returned predictions
                for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)

                # indicate that the request was a success
                data["success"] = True


                data['diag_name']=random.choice(diag_list)
                data['diag_type']=random.choice(diag_type)

                #similar Images
                #random number from 1 to 10
                r_num = random.randint(1,5)
                tmp_im='sim_' + str(r_num) + '.jpg'
                data['sim_1']=os.path.join(app.config['STATIC_FOLDER'], tmp_im)

                r_num = random.randint(6,10)
                tmp_im='sim_' + str(r_num)  + '.jpg'
                data['sim_2']=os.path.join(app.config['STATIC_FOLDER'], tmp_im)

                r_num = random.randint(11,15)
                tmp_im='sim_' + str(r_num)  + '.jpg'
                data['sim_3']=os.path.join(app.config['STATIC_FOLDER'], tmp_im)
                
                print(data)    
        return render_template("index.html",skin_data=data)

        #return jsonify(data)
        #return '''
        #<!doctype html>
        #<title>Skin Image Diagnosis</title>
        #<h1>Skin Cancer Image Classification Project</h1>
        #<h3>Northwestern Data Science Bootcamp</h3>
        #<br>
        #<h1>Image Classification Diagnosis</h1>
        #<div class="thumbnail"><img src= "{{data.img_src}}"></div>
        #<h3>Predicted Diagnosis:</h3>
        #<h2>Melonoma: Benign</h2> 
        #<br>
        #<h2>Similar Images found in database</h2>
        #'''

        #return render_template("index.html",mars_data=data)    
    
    #return template and data
    #return render_template("index.html")    
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1 style="color:red">Skin Cancer Image Classification Project</h1>
    <p>Northwestern Data Science Bootcamp</p>
    <br>
    <br>
    <h1>Upload Image File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
    

if __name__ == "__main__":
    app.run(debug=True)
