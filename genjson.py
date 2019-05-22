import os
import json
import numpy as np

X_train = []
y_train = []
label_count = 0
nolabel_count = 0
i = 0
label = 'arrow_backward'
for i in range(1, 58000):
    i = i+1
    if(i%1000==0):
        print('label count:{}\tNo label count:{}'.format(label_count, nolabel_count))

    #path to dataset
    path = '/work3/s182091/rico/semantic_annotations/{}.json'.format(i)
    if(os.path.isfile(path)):
        with open(path) as json_file:
            legend = json.load(json_file)
            if("'iconClass': '{}'".format(label) in str(legend['children'])):
                #limit number of samples to obtain balanced dataset
                if(label_count < 5000):
                    X_train.append(i)
                    label_count += 1
                    y_train.append(1)
            elif(nolabel_count < 5000):
                    X_train.append(i)
                    nolabel_count += 1
                    y_train.append(0)

print('label count:{}\tNo label count:{}'.format(label_count, nolabel_count))

X_train = np.array(X_train)
y_train = np.array(y_train)

#shuffle dataset so validation data from the end is also balanced
s = np.arange(y_train.shape[0])
np.random.shuffle(s)

X_train = X_train[s]
y_train = y_train[s]

#print few examples for double checking
for i in range(10):
    print('For photo {}, label is {}'.format(X_train[i], y_train[i]))

with open('X_train_arrow.txt', 'w+') as outfile:  
    json.dump(X_train.tolist(), outfile)

with open('y_train_arrow.txt', 'w+') as outfile:  
    json.dump(y_train.tolist(), outfile)















