

class Base_model(object):
    def __init__(self, X_train, y_train, X_valid, y_valid,params, 
                iter, random_state = 0):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.iter = iter
        self.random_state = random_state
        self.space = params