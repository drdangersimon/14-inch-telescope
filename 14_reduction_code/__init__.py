import src
import numpy as np


"test"
class reduce_daemon(object):
    '''Reduction daemon to do auto reduction. Checks a directory for new files
and will sort then reduce them automatically. Once reduction is complete,
will do photometry, and add postage stamps to a database'''

    def __init__(self,watch_path,cal_path,database_path):
        #initalize
        self.watch_path = watch_path
        self.cal_path = cal_path
        self.database_path = database_path



if __name__ == '__main__':

    '''comand lin running'''
