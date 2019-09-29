import numpy as np
import keras
from torch.utils.data import Dataset
from basic_transformer import utils as local_util


class TextDataset(Dataset):
    """
    Text Dataset
    """
    def __init__(self, 
                 df, 
                 num_words, 
                 text_column, 
                 label_column, 
                 label_mapping, 
                 max_seq_len):
        self.df = df
        self.num_words = num_words
        self.text_column = text_column
        self.label_column = label_column
        self.label_mapping = label_mapping
        self.max_seq_len = max_seq_len
        
        # fit tokenizer
        self.tokenizer = keras.preprocessing.text.Tokenizer(num_words=self.num_words)
        self.tokenizer.fit_on_texts(self.df[text_column])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # seq
        txt = self.df.iloc[idx][self.text_column]
        seq = self.tokenizer.texts_to_sequences([txt])[0]
        seq = local_util.data.fix_seq_len(seq, self.max_seq_len)
        seq = np.array(seq)
        
        # label
        label = self.df.iloc[idx][self.label_column]
        label = self.label_mapping[label]
        
        return {'seq': seq, 'label': label}
