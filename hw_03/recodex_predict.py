import pickle
def recodex_predict(data):
    # The `data` is a Numpy array containt test se

    with open("linear_regression_competition.model", "rb") as model_file:
        model = pickle.load(model_file)

    predictions = model.predict(data)
    return predictions
