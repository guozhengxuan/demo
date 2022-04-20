import glob

from sa_model.model_wrapper import Wrapper_Model
from cfg import *


class ClassificationPredictor:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_path = glob.glob(os.path.join(SA_BEST_MODEL, '*.ckpt'))[0]
        self.model = Wrapper_Model.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = self.model.tokenizer

    def predict(self, payload):
        inputs = self.tokenizer.encode_plus(payload['text'], return_tensors='pt')
        outputs = self.model(**inputs)
        logits = outputs['logits']
        predicts = torch.argmax(logits)
        return int(predicts)


if __name__ == '__main__':
    # test
    pd = ClassificationPredictor()
    pl = {'text': "哈哈哈笑死我了"}
    pd.predict(pl)
