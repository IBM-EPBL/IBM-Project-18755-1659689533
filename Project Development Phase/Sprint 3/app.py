from flask import Flask, render_template,request
from feature import FeatureExtraction
import numpy as np
import joblib
import requests

app = Flask(__name__)

@app.route('/',methods=['GET'])
def hello():
    return render_template('index.html')

@app.route("/", methods=["POST"])
def index(): 

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
        API_KEY = "3F37hSWwvdU4e8YiYw1sywqPxaTRoa8OXV0z88TT9UEJ"
        token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
        API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
        mltoken = token_response.json()["access_token"]

        header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

        url = request.form['url']
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30) 
        print(x.tolist())
        # NOTE: manually define and pass the array(s) of values to be scored in the next line
        payload_scoring = {"input_data": [{"field": [["index",	"having_IPhaving_IP_Address", "URLURL_Length",	"Shortining_Service","having_At_Symbol",	"double_slash_redirecting",	"Prefix_Suffix",	"having_Sub_Domain","SSLfinal_State","SSLfinal_State",	"Domain_registeration_length","Favicon","port",	"HTTPS_token",	"Request_URL",	"URL_of_Anchor",	"Links_in_tags",	"SFH",	"Submitting_to_email",	"Abnormal_URL",	"Redirect",	"on_mouseover",	"RightClick",	"popUpWidnow",	"Iframe",	"age_of_domain",	"DNSRecord",	"web_traffic",	"Page_Rank",	"Google_Index",	"Links_pointing_to_page",	"Statistical_report"
        ]], "values": x.tolist()}]}

        response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/81bdb427-49b7-4253-b4a1-368940ece95b/predictions?version=2022-11-13', json=payload_scoring,
        headers={'Authorization': 'Bearer ' + mltoken})

        response_text = response_scoring.json()
        y_pred = response_text['predictions'][0]['values'][0][0]
        y_pro_phishing = response_text['predictions'][0]['values'][0][1][0]
        y_pro_non_phishing = response_text['predictions'][0]['values'][0][1][1]
        # gbc = joblib.load('model.pkl')
        # y_pred =gbc.predict(x)[0]
        # #1 is safe       
        # #-1 is unsafe
        # y_pro_phishing = gbc.predict_proba(x)[0,0]
        # y_pro_non_phishing = gbc.predict_proba(x)[0,1]

        if(y_pred == 0 ):
            url = "https://"+url 
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url )
if __name__ =='__main__':
    app.run(port=5500,debug=True)
