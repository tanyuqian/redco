class Dataset:
    def get_examples(self, split):
        raise NotImplementedError

    def get_size(self, split):
        raise NotImplementedError
