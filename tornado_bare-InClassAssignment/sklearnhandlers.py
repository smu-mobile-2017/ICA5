#!/usr/bin/python

from pymongo import MongoClient
import tornado.web

from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options

from basehandler import BaseHandler

from sklearn.neighbors import KNeighborsClassifier
import pickle
from bson.binary import Binary
import json
import numpy as np

class PrintHandlers(BaseHandler):
    def get(self):
        '''Write out to screen the handlers used
        This is a nice debugging example!
        '''
        self.set_header("Content-Type", "application/json")
        self.write(self.application.handlers_string.replace('),','),\n'))

class UploadLabeledDatapointHandler(BaseHandler):
    def post(self):
        '''Save data point and class label to database
        '''
        data = json.loads(self.request.body.decode("utf-8"))

        vals = data['feature']
        fvals = [float(val) for val in vals]
        label = data['label']
        sess  = data['dsid']

        dbid = self.db.labeledinstances.insert(
            {"feature":fvals,"label":label,"dsid":sess}
            );
        self.write_json({"id":str(dbid),
            "feature":[str(len(fvals))+" Points Received",
                    "min of: " +str(min(fvals)),
                    "max of: " +str(max(fvals))],
            "label":label})

class RequestNewDatasetId(BaseHandler):
    def get(self):
        '''Get a new dataset ID for building a new dataset
        '''
        a = self.db.labeledinstances.find_one(sort=[("dsid", -1)])
        if a == None:
            newSessionId = 1
        else:
            newSessionId = float(a['dsid'])+1
        self.write_json({"dsid":newSessionId})

def newModel(self, dsid):
    # create feature vectors from database
    f=[]
    for a in self.db.labeledinstances.find({"dsid":dsid}):
        f.append([float(val) for val in a['feature']])

    # create label vector from database
    l=[]
    for a in self.db.labeledinstances.find({"dsid":dsid}):
        l.append(a['label'])

    # fit the model to the data
    c1 = KNeighborsClassifier(n_neighbors=1)
    acc = -1
    if l:
        c1.fit(f,l) # training
        lstar = c1.predict(f)
        self.clf[dsid] = c1

        acc = sum(lstar==l)/float(len(l))
        bytes = pickle.dumps(c1)

        self.db.models.update(
            {"dsid": dsid},
            {"$set":
                {
                    "model": Binary(bytes)
                }
            },
            upsert=True
        )
    return acc

class UpdateModelForDatasetId(BaseHandler):
    def get(self):
        '''Train a new model (or update) for given dataset ID
        '''
        dsid = self.get_int_arg("dsid",default=0)

        acc = newModel(self, dsid)

        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json({"resubAccuracy":acc})

class PredictOneFromDatasetId(BaseHandler):
    def post(self):
        '''Predict the class of a sent feature vector
        '''
        data = json.loads(self.request.body.decode("utf-8"))

        vals = data['feature'];
        fvals = [float(val) for val in vals];
        fvals = np.array(fvals).reshape(1, -1)
        dsid  = data['dsid']

        # load the model from the database (using pickle)
        # we are blocking tornado!! no!!

        if dsid not in self.clf:
            print('Loading Model From DB')
            modelPersistence = self.db.models.find_one({"dsid":dsid})
            if modelPersistence:
                self.clf[dsid] = pickle.loads(modelPersistence['model'])
            else:
                newModel(self, dsid)

        predLabel = self.clf[dsid].predict(fvals);
        self.write_json({"prediction":str(predLabel)})
