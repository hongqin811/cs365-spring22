import pandas as pd
import numpy as np

def load(path):
    df = None
    '''YOUR CODE HERE'''
    df = pd.read_csv(path)

    '''END'''
    return df

def prior(df):
    ham_prior = 0
    spam_prior =  0
    '''YOUR CODE HERE'''
    
    # if label == 0, then ham, else spam
    length = df['label_num'].shape[0]
    for i in df['label_num']:
        if i == 0:
            ham_prior += 1
        else:
            spam_prior += 1
        
    # prior divide by total length to get probability
    ham_prior /= length
    spam_prior /= length

    '''END'''
    return ham_prior, spam_prior

def likelihood(df):
    ham_like_dict = {}
    spam_like_dict = {}
    '''YOUR CODE HERE'''
    
    # seperate df into 2 different df
    ham_df = df[df['label_num'] == 0]
    spam_df = df[df['label_num'] == 1]
    
    # count the number of email of 2 classes
    num_ham = ham_df.shape[0]
    num_spam = spam_df.shape[0]
    
    # start building ham_like_dict
    for i in ham_df['text']:
        text_list = i.split()
        # remove duplicate
        text_list = list(dict.fromkeys(text_list))
        for j in text_list:
            if ham_like_dict.get(j) == None:
                ham_like_dict[j] = (1/num_ham)
            else:
                ham_like_dict[j] += (1/num_ham)
                
    # start building spam_like_dict
    for i in spam_df['text']:
        text_list = i.split()
        # remove duplicate
        text_list = list(dict.fromkeys(text_list))
        for j in text_list:
            if spam_like_dict.get(j) == None:
                spam_like_dict[j] = (1/num_spam)
            else:
                spam_like_dict[j] += (1/num_spam)

    '''END'''

    return ham_like_dict, spam_like_dict

def predict(ham_prior, spam_prior, ham_like_dict, spam_like_dict, text):
    '''
    prediction function that uses prior and likelihood structure to compute proportional posterior for a single line of text
    '''
    #ham_spam_decision = 1 if classified as spam, 0 if classified as normal/ham
    ham_spam_decision = None


    '''YOUR CODE HERE'''
    
    #ham_posterior = posterior probability that the email is normal/ham
    ham_posterior = np.log10(ham_prior)
    
    text_list = text.split()
    
    for i in text_list:
        likelihood = ham_like_dict.get(i)
        if likelihood != None:
            ham_posterior += np.log10(likelihood)

    #spam_posterior = posterior probability that the email is spam
    spam_posterior = np.log10(spam_prior)
    
    text_list = text.split()

    for i in text_list:
        likelihood = spam_like_dict.get(i)
        if likelihood != None:
            spam_posterior += np.log10(likelihood)
            
    if ham_posterior > spam_posterior:
        ham_spam_decision = 0  
    else:
        ham_spam_decision = 1
    

    '''END'''
    return ham_spam_decision 


def metrics(ham_prior, spam_prior, ham_dict, spam_dict, df):
    '''
    Calls "predict"
    '''
    
    hh = 0 #true negatives, truth = ham, predicted = ham
    hs = 0 #false positives, truth = ham, pred = spam
    sh = 0 #false negatives, truth = spam, pred = ham
    ss = 0 #true positives, truth = spam, pred = spam
    num_rows = df.shape[0]
    for i in range(num_rows):
        roi = df.iloc[i,:]
        roi_text = roi.text
        roi_label = roi.label_num
        guess = predict(ham_prior, spam_prior, ham_dict, spam_dict, roi_text)
        if roi_label == 0 and guess == 0:
            hh += 1
        elif roi_label == 0 and guess == 1:
            hs += 1
        elif roi_label == 1 and guess == 0:
            sh += 1
        elif roi_label == 1 and guess == 1:
            ss += 1
    
    acc = (ss + hh)/(ss+hh+sh+hs)
    precision = (ss)/(ss + hs)
    recall = (ss)/(ss + sh)
    return acc, precision, recall
    
if __name__ == "__main__":
    '''YOUR CODE HERE'''
    #this cell is for your own testing of the functions above
    
    df = load("./TEST_balanced_ham_spam.csv")
    
    ham_prior, spam_prior = prior(df)
    ham_dict, spam_dict = likelihood(df)
    
    
    acc, precision, recall = metrics(ham_prior, spam_prior, ham_dict, spam_dict, df)
    
    print("acc\t =" + str(acc))
    print("precision=" + str(precision))
    print("recall\t = " + str(recall))
    
    