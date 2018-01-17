import nltk
from nltk.model.counter import NgramModelVocabulary
from nltk.model import count_ngrams
from collections import Counter
from nltk.model.counter import NgramCounter
import javalang
import glob
from tqdm import tqdm_notebook



class RepoLanguageModel:
    
    def __init__(self,path):        
        self.path = path
        self.corpus = []
        
    def get_all_repo_files(self):
        
        files = []
        for filename in glob.iglob(self.path + '**/*.java', recursive=True):
            files.append(filename)
        return files
        
    def create_corpus(self, files):
        
        parsed_files = []
        
        print('Bad files: ')
        for filename in tqdm_notebook(files):
            with open(filename, 'r') as file:
                text = file.read()            
                stoptrans = str.maketrans('', '', '#\'`\\"')
                try:
                    parsed_files.append([t.value for t in javalang.tokenizer.tokenize(text.translate(stoptrans))])
                except:
                    print(filename)
                    
        corpus = []
        for file in tqdm_notebook(parsed_files):
            line = []
            for i in range(len(file)):
                if file[i] != ';':
                    line.append(file[i])
                else:
                    corpus.append(line)
                    line = []
        self.corpus = corpus
        self.vocabulary = NgramModelVocabulary(1, [j for i in corpus for j in i])
        return 
    
    def train_ngrams(self, corpus, n):
        self.ngrams = NgramCounter(n, self.vocabulary)
        self.ngrams.train_counts(self.corpus)
        return
    
    
    def create_model(self, model_class, n):
        print('create corpus')
        self.create_corpus(self.get_all_repo_files())
        print('train ngrams')
        self.train_ngrams(self.corpus, n)
        self.model = model_class(self.ngrams)
        print('model is ready')