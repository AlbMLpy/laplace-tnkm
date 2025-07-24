from collections import defaultdict

from .evaluation import norm_frob
from .matrix_operations import ten3tovec
from .general_functions import update_results_dict

class SimpleTracker:
    def __init__(self, x_train, y_train, beta_e, gamma_w, loss, grad_w):
        self.res_dict = defaultdict(list)
        self.x_train = x_train
        self.y_train = y_train
        self.beta_e = beta_e
        self.gamma_w = gamma_w
        self.loss = loss
        self.grad_w = grad_w

    def track(self, w_ten, kd, fmap):
        w_vec, w_shape = ten3tovec(w_ten), w_ten.shape
        train_loss = self.loss(
            w_vec, kd, self.x_train, self.y_train, 
            fmap, self.gamma_w, self.beta_e, w_shape
        ).item()
        grad_norm = norm_frob(
            self.grad_w(
                w_vec, kd, self.x_train, self.y_train, 
                fmap, self.gamma_w, self.beta_e, w_shape
            )
        ).item()
        update_results_dict(
            self.res_dict, loss=train_loss, grad_norm=grad_norm
        )
