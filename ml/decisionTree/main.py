import pickle, json


lookup_labels_path   =  "ml/decisionTree/assets/lookup_labels.pkl"
decision_model_path  =  "ml/decisionTree/assets/decision_model.sav"
feature_encoder_path =  "ml/decisionTree/assets/feature_encoder.pkl"

# load feature from disk
tam_trang_encoder, tinh_trang_encoder, the_trang_encoder, khau_vi_encoder, thoi_diem_encoder = pickle.load(open(feature_encoder_path, 'rb'))
# load the model from disk
clf = pickle.load(open(decision_model_path, 'rb'))
# load the lookup labels from disk
lookup_labels = pickle.load(open(lookup_labels_path, 'rb'))

def inference(input_demands):
  '''
    input_demands: [tam_trang string, tinh_trang string,  the_trang string, khau_vi string, thoi_diem string ]
  '''
  input = list(zip(tam_trang_encoder.transform([input_demands[0]]),
        tinh_trang_encoder.transform([input_demands[1]]),
        the_trang_encoder.transform([input_demands[2]]),
        khau_vi_encoder.transform([input_demands[3]]),
        thoi_diem_encoder.transform([input_demands[4]])))
  y_predict = clf.predict(input)[0]
  try:
    return json.loads(lookup_labels.get(y_predict, None))
  except:
    return lookup_labels.get(y_predict, None)