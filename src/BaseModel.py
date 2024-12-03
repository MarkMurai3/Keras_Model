from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def compile_model(self, optimizer, loss, metrics):
        pass

    @abstractmethod
    def train_model(self, x_train, y_train, validation_data, epochs, batch_size, save_plot_path):
        pass

    @abstractmethod
    def evaluate_model(self, x_test, y_test):
        pass

    @abstractmethod
    def save_model(self, filepath):
        pass

    @staticmethod
    def dynamic_cast(obj, target_class):
        if isinstance(obj, target_class):
            return obj
        else:
            raise TypeError(f"Cannot cast object to {target_class.__name__}. Object is not of the correct type.")
