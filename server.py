#------------------------ importing modules -----------------------------------

from flask import Flask, render_template, jsonify, request
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import recom
import json

#------------------------- configs and vars -----------------------------------

app = Flask(__name__)
disease_type = ["Bacterial leaf blight", "Brown spot", "Leaf smut"]
features=['temp','rainfall','ph','irrigation','yield','sowing_time','soiltype','maturity']
inp=dict()

#-------------------------- loading models ------------------------------------

with open('./model_config.json') as file:
    config=file.read()
file.close()
model=keras.models.model_from_json(config)
model.load_weights('./weights.h5')

#-------------------------- routes from here ----------------------------------

# bad request page error handeler
@app.errorhandler(404)
def not_found(e):
  return jsonify({'error' : str(e)})

#------------------------->>>>>>>>>>>>>>>>>>>>>--------------------------------
# home page
@app.route("/")
def welcome():
    return render_template("welcomepage.html")

# high priority page
@app.route("/detectdiseasepage")
def detectdiseasepage():
    return render_template("diseasedetection.html",status='No image uploaded!')


@app.route("/cropprediction1")
def cropprediction1():
    return render_template("cropprediction1.html")


@app.route("/cropprediction2")
def cropprediction2():
    return render_template("cropprediction2.html")


@app.route("/cropprediction3",methods=['post'])
def cropprediction3():
    data=request.form
    inp.update(dict(data))

    soiltype=[0]*5
    ind=int(inp['soiltype'])
    soiltype[ind]=1
    inp['soiltype']=soiltype

    return render_template("cropprediction3.html")


@app.route("/cropprediction4",methods=['post'])
def cropprediction4():
    data=request.form
    inp.update(dict(data))
    return render_template("cropprediction4.html")


@app.route("/cropprediction5",methods=['post'])
def cropprediction5():
    data=request.form
    inp.update(dict(data))

    maturity=[0]*3
    ind=int(inp['maturity'])
    maturity[ind]=1
    inp['maturity']=maturity

    inp['sowing_time']=int(inp['day'])+30*int(inp['month'])
    return render_template("cropprediction5.html")

@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")


@app.route("/studyarea")
def studyarea():
    return render_template("studyarea.html")


@app.route("/technologyused")
def technologyused():
    return render_template("technologyused.html")


@app.route("/whatwedo")
def whatwedo():
    return render_template("whatwedo.html")
# -------------------------->>>>>>>>>>>>>>>>>>>>>>-----------------------------




#--------------------------->>>>>>>>>>>>>>>>>>>>>>>----------------------------
# detect disease of the crop
@app.route("/detectdisease", methods=["POST"])
def detect_disease():
    print(request)
    img = request.files["image"].read()                         # getting the image fill
    npimg = np.fromstring(img, np.uint8)                        # convert string file obj. to np array
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)             # convert np array to image
    re_img = cv2.resize(img, (200, 200)) / 255.0                        # resize the image to desired size
    re_tensor = re_img.reshape(1, 200, 200, 3)                  # reshape the image

    one_hot = model.predict(re_tensor)                              # predictions predictions predictions
    print(one_hot)
    out=disease_type[int(np.argmax(one_hot, axis=1))]
    string='Disease detected!'
    return render_template('./diseasedetection.html',output=str(out),status=string)



# predict crops
@app.route("/croprecommendation")
def croppredict():
    data=[]
    for feature in features:
        data.append(inp[feature])

    result=recom.get_recommendations(data,3)
    return json.dumps(result)
# ---------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>----------------------------------------


# main
if __name__ == "__main__": app.run(debug = True)
