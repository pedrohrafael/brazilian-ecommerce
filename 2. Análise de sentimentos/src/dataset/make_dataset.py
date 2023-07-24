import pandas as pd
import numpy as np

def load_reviews(path):
    df = pd.read_csv(path)
    
    # concatenate title and message comments
    df['review_comments'] = (df.review_comment_title.astype(str) +' '+ df.review_comment_message.astype(str)).str.replace('nan', '')
    df['review_comments'] = np.where(df['review_comments']==' ', None, df['review_comments'])
    
    # target by review score - negative: 1, 2, 3 | positive: 4, 5
    df['target'] = np.where(df.review_score > 3, 1, 0)
    
    # remove null comments
    df = df[df.review_comments.notnull()].reset_index()[['review_comments', 'target']]
    
    return df