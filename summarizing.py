from transformers import pipeline

class Summarizer():
    def __init__(self,long_text):
        self.long_text = long_text


    def summary(self):
        text = self.long_text
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        result = summarizer(text, max_length=130, min_length=30, do_sample=False)
        text = result[0]['summary_text']
        return text
    



                    


