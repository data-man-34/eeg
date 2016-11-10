import cPickle
import numpy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from astropy.table import Table, join
#videolist=numpy.genfromtxt(fname="metadata_csv/video_list.csv",skip_header=1,usecols=(1,2),delimiter=',',dtype=[int,basestring])
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split




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
data=numpy.empty([0,feature])
for i in xrange(1,33):
    if(i<10):
        x=cPickle.load(open(str1+"0"+str(i)+str2, 'rb'))
    else:
        x=cPickle.load(open(str1+str(i)+str2, 'rb'))
    # print x['data']
    # break
    tmp = numpy.hstack((numpy.min(x["data"],axis=2),numpy.max(x["data"],axis=2),numpy.mean(x["data"],axis=2)))
    data = numpy.vstack((data,tmp))
# print(data.shape)
data=pd.DataFrame(data)
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