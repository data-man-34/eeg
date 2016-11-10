import cPickle
import sys
import os
import numpy as np
import pandas as pd
from math import *
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
# from astropy.table import Table, join
#videolist=numpy.genfromtxt(fname="metadata_csv/video_list.csv",skip_header=1,usecols=(1,2),delimiter=',',dtype=[int,basestring])
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
def corr(data,type_corr):
    C = np.array(data.corr(type_corr))
    C[np.isnan(C)] = 0
    C[np.isinf(C)] = 0
    w,v = np.linalg.eig(C)
    #print(w)
    x = np.sort(w)
    x = np.real(x)
    return x


####  feature addition
def cal_feature(data):
    fs = 128
    eegData = data.transpose()
    [nt, nc] = eegData.shape
    print((nt, nc))
    # subsampLen = floor(fs * 60)
    # numSamps = int(floor(nt / subsampLen));  # Num of 1-min samples
    # sampIdx = range(0, (numSamps + 1) * subsampLen, subsampLen)
    # print(sampIdx)
    feat = []  # Feature Vector
    # for i in range(1, numSamps + 1):
    # print('processing file {} epoch {}'.format(file_name, i))
    epoch = eegData

    # compute Shannon's entropy, spectral edge and correlation matrix
    # segments corresponding to frequency bands
    lvl = np.array([0.1, 4, 8, 12, 30, 70, 180])  # Frequency levels in Hz
    lseg = np.round(nt / fs * lvl).astype('int')
    D = np.absolute(np.fft.fft(epoch, n=lseg[-1], axis=0))
    D[0, :] = 0  # set the DC component to zero
    D /= D.sum()  # Normalize each channel

    dspect = np.zeros((len(lvl) - 1, nc))
    for j in range(len(dspect)):
        dspect[j, :] = 2 * np.sum(D[lseg[j]:lseg[j + 1], :], axis=0)

    # Find the shannon's entropy
    spentropy = -1 * np.sum(np.multiply(dspect, np.log(dspect)), axis=0)

    # Find the spectral edge frequency
    sfreq = fs
    tfreq = 40
    ppow = 0.5

    topfreq = int(round(nt / sfreq * tfreq)) + 1
    A = np.cumsum(D[:topfreq, :])
    B = A - (A.max() * ppow)
    spedge = np.min(np.abs(B))
    spedge = (spedge - 1) / (topfreq - 1) * tfreq

    # Calculate correlation matrix and its eigenvalues (b/w channels)
    data = pd.DataFrame(data=epoch)
    type_corr = 'pearson'
    lxchannels = corr(data, type_corr)

    # Calculate correlation matrix and its eigenvalues (b/w freq)
    data = pd.DataFrame(data=dspect)
    lxfreqbands = corr(data, type_corr)

    # Spectral entropy for dyadic bands
    # Find number of dyadic levels
    ldat = int(floor(nt / 2.0))
    no_levels = int(floor(log(ldat, 2.0)))
    seg = floor(ldat / pow(2.0, no_levels - 1))

    # Find the power spectrum at each dyadic level
    dspect = np.zeros((no_levels, nc))
    for j in range(no_levels - 1, -1, -1):
        dspect[j, :] = 2 * np.sum(D[int(floor(ldat / 2.0)) + 1:ldat, :], axis=0)
        ldat = int(floor(ldat / 2.0))

    # Find the Shannon's entropy
    spentropyDyd = -1 * np.sum(np.multiply(dspect, np.log(dspect)), axis=0)

    # Find correlation between channels
    data = pd.DataFrame(data=dspect)
    lxchannelsDyd = corr(data, type_corr)

    # Fractal dimensions
    no_channels = nc
    # fd = np.zeros((2,no_channels))
    # for j in range(no_channels):
    #    fd[0,j] = pyeeg.pfd(epoch[:,j])
    #    fd[1,j] = pyeeg.hfd(epoch[:,j],3)
    #    fd[2,j] = pyeeg.hurst(epoch[:,j])

    # [mobility[j], complexity[j]] = pyeeg.hjorth(epoch[:,j)
    # Hjorth parameters
    # Activity
    activity = np.var(epoch, axis=0)
    # print('Activity shape: {}'.format(activity.shape))
    # Mobility
    mobility = np.divide(
        np.std(np.diff(epoch, axis=0)),
        np.std(epoch, axis=0))
    # print('Mobility shape: {}'.format(mobility.shape))
    # Complexity
    complexity = np.divide(np.divide(
        # std of second derivative for each channel
        np.std(np.diff(np.diff(epoch, axis=0), axis=0), axis=0),
        # std of second derivative for each channel
        np.std(np.diff(epoch, axis=0), axis=0))
        , mobility)
    # print('Complexity shape: {}'.format(complexity.shape))
    # Statistical properties
    # Skewness
    sk = skew(epoch)

    # Kurtosis
    kurt = kurtosis(epoch)

    # compile all the features
    feat = np.concatenate((feat,
                           spentropy.ravel(),
                           spedge.ravel(),
                           lxchannels.ravel(),
                           lxfreqbands.ravel(),
                           spentropyDyd.ravel(),
                           lxchannelsDyd.ravel(),
                           # fd.ravel(),
                           activity.ravel(),
                           mobility.ravel(),
                           complexity.ravel(),
                           sk.ravel(),
                           kurt.ravel()
                           ))

    return feat


###


####extracting data
videolist=pd.read_csv("metadata_csv/video_list.csv",usecols=[1,2],delimiter=',')
videolist.dropna(axis=0,how='any',inplace=True)
# print(videolist)
#rating=numpy.genfromtxt(fname="metadata_csv/participant_ratings.csv",skip_header=1,usecols=(2),delimiter=',',dtype=[int])
rating=pd.read_csv("metadata_csv/participant_ratings.csv",usecols=[2],delimiter=',')
# print(rating)
#tag=join(rating,videolist,join_type='inner')
tag=rating.merge(videolist,how='left')
# print(tag)
# tag.dropna(axis=0,how='any',inplace=True)
# print tag
str1="data_preprocessed_python/s"
str2=".dat"
feature = 120
data=np.empty([0,401])
for i in xrange(1,33):
    if(i<10):
        x=cPickle.load(open(str1+"0"+str(i)+str2, 'rb'))
    else:
        x=cPickle.load(open(str1+str(i)+str2, 'rb'))
    # print x['data']
    # break
    for row in x["data"]:
        data=np.vstack((data,cal_feature(row)))
        print(data.shape)
    #tmp = np.hstack((np.min(x["data"],axis=2),np.max(x["data"],axis=2),np.mean(x["data"],axis=2)))
    #data = np.vstack((data,tmp))
#data=pd.DataFrame(data)

#---------***--------*



###   training adn testing part
data=pd.concat([data,tag['Lastfm_tag']],axis=1)
print data
# data=numpy.column_stack((data,tag["Lastfm_tag"]))
# print(data.shape)
# print(data[0])
# data=pd.DataFrame(data)
data.dropna(axis=0, how='any',inplace=True)
print (data)
# df = pd.DataFrame(numpy.random.randn(544, 2))
# msk = numpy.random.rand(len(df)) < 0.8
outcome=data['Lastfm_tag']
data.drop(['Lastfm_tag'],inplace=True, axis=1)
train,test,traintag,testtag=train_test_split(data,outcome,test_size=0.3)
# train = data[msk]
# test = data[~msk]

# traintag=train.ix[:,feature]
# testtag=test.ix[:,feature]
# train.drop(train.columns[[feature]], axis=1,inplace=True)
# test.drop(test.columns[[feature]], axis=1,inplace=True)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(train, traintag)

# print(rf.predict(test))
print accuracy_score(testtag,rf.predict(test))
# print(testtag)