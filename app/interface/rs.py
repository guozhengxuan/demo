import json
from app.interface import request, app
import numpy as np
from app.model.predict import ClassificationPredictor
from flask import jsonify

label2str = {
    "0": "喜悦",
    "1": "愤怒",
    "2": "厌恶",
    "3": "低落",
}

predictor = ClassificationPredictor()


@app.route('/classify/sa', methods=['POST'])
def sa_classify():
    params = json.loads(request.get_data(as_text=True))
    params['text'] = params['text'].replace('\t', ' ').replace('\n', ' ').replace(' ', '')
    pred = predictor.predict(params)
    res = {
        'label': label2str[str(pred)]
    }
    return jsonify(res)


if __name__ == "__main__":
    # port: -》 8897
    """ TODO 更换port """
    app.run(host='127.0.0.1', debug=True)
