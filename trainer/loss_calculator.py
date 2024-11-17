class LossCalculator:
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn  

    def compute_loss(self, predicted_noise, generated_noise):
        return self.loss_fn(predicted_noise, generated_noise)