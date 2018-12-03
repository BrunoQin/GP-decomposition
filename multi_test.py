import gpflow as gp
import tensorflow as tf
import numpy as np
from concurrent import futures
from threading import Lock


def _predict(x,y,lock,graph = None):
    # lock is created in parent thread and passed to all children
    with tf.Session(graph=tf.Graph()) as sess:
        #Lock on the AutoBuild global
        lock.acquire()
        try:
            with gp.defer_build():
                kern = gp.kernels.RBF(1)
                model = gp.models.GPR(x, y, kern)
        finally:
            lock.release()
        model.compile()
        # training etc here
        ystar,varstar = model.predict_f(x)
        gp.train.ScipyOptimizer().minimize(model)
        ystar,varstar = model.predict_f(x)
    return ystar, varstar


def async(X,Y, max_workers):
    ###
    # Possibly do some GP optimization etc.
    # then run some gpflow operations in threads
    jobs = []
    graph = tf.Graph
    lock = Lock()
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for id,(x,y) in enumerate(zip(X,Y)):
            jobs.append(executor.submit(_predict, x,y,lock, graph=graph))
        futures.wait(jobs)
        results = [j.result() for j in jobs]


if __name__ == '__main__':
    async(np.random.normal(size=[20,100,1]),
          np.random.normal(size=[20,100,1]),
          8)
