from django.shortcuts import render
import pickle
from sklearn.preprocessing import FunctionTransformer

def home(request):
    prediction_result = None  # To pass to the template
    if request.method == "POST":
        fst = FunctionTransformer(validate=False)  # Disable validation for single strings
        text1 = request.POST.get("text1", "")
        
        # Transform the input text
        texts1 = fst.transform([text1])  # Wrap in list

        # Load the TF-IDF vectorizer
        with open('C:/users/kunal/downloads/mm/spam_detector/pickle_files/new_vectorizer.pkl', 'rb') as f:
            tfidf_load = pickle.load(f)

        # Load the trained model
        with open('C:/users/kunal/downloads/mm/spam_detector/pickle_files/new_model.pkl', 'rb') as f:
            
            model = pickle.load(f)

        # Transform and predict
        inputtrans = tfidf_load.transform(texts1)
        prediction = model.predict(inputtrans)

        # Set prediction result for use in the template
        if prediction[0] == 1:
            prediction_result = "Spam"
        else:
            prediction_result = "Not Spam"

    return render(request, 'app/templates/index.html', {'prediction': prediction_result})
