import datetime
import math
import numpy
import random
import operator
from collections import Counter
from functools import reduce



def weibullpdf(data, scale, shape):
    """# homegrown weibull probability function
    scipy.stats.exonweib.pdf gives different result from R dweibull function"""
    return [(shape / scale) * ((x / scale) ** (shape - 1)) * math.exp(-1 * (x / scale) ** shape) for x in data]


def matmult4(m, v):
    """matrix-vector multiplier from http://code.activestate.com/recipes/121574-matrix-vector-multiplication/
    equivalent to %*% in R"""
    return [reduce(operator.add, map(operator.mul, r, v)) for r in m]


class WindResource(object):
    def __init__(self, mean_speed=9.0, max_speed=30.0, shape=2, scale=1.2, n_states=30,
                 start_time=datetime.datetime(2017, 1, 1), time_delta=1):
        self.mean_speed = mean_speed
        self.max_speed = max_speed
        self.n_states = n_states
        self.time = start_time
        # self.hist_length = hist_length
        # Receive the time delta in minutes. Transform into seconds
        self.time_delta = time_delta*60

        # setup matrix
        n_rows = n_states
        n_columns = n_states
        self.l_categ = float(max_speed) / float(n_states)  # position of each state

        # weibull parameters
        self.weib_shape = shape
        self.weib_scale = scale

        # Vector of wind speeds
        self.WindSpeed = numpy.arange(self.l_categ / 2.0, float(max_speed) + self.l_categ / 2.0, self.l_categ)

        # distribution of probabilities, normalised
        fdpWind = weibullpdf(self.WindSpeed, self.weib_scale, self.weib_shape)
        fdpWind = fdpWind / sum(fdpWind)

        # decreasing function
        G = numpy.empty((n_rows, n_columns,))
        for x in range(n_rows):
            for y in range(n_columns):
                G[x][y] = 2.0 ** float(-abs(x - y))

        # Initial value of the P matrix
        P0 = numpy.diag(fdpWind)

        # Initital value of the p vector
        p0 = fdpWind

        # below comment from R source code:
        # "The iterative procedure should stop when reaching a predefined error.
        # However, for simplicity I have only constructed a for loop. To be improved!"
        P, p = P0, p0
        for i in range(10):
            r = matmult4(P, matmult4(G, p))
            r = r / sum(r)
            p = p + 0.5 * (p0 - r)
            P = numpy.diag(p)

        N = numpy.diag([1.0 / i for i in matmult4(G, p)])
        MTM = matmult4(N, matmult4(G, P))
        self.MTMcum = numpy.cumsum(MTM, 1)

        # initialise series
        self.state = 0
        self.states_series = []
        self.speed_series = []
        self.power_series = []
        self.randoms1 = []
        self.randoms2 = []

        # tick over to first value (decrement time accordingly)
        self.time = self.time + datetime.timedelta(seconds=-self.time_delta)
        self.getNext()

    # show current value without incrementing
    def getCurrent(self):
        wind_counter = Counter([int(round(x, 0)) for x in self.speed_series])
        # return {'time': self.time,
        #         'data': {
        #             'wind_speed': self.speed_series[-1],
        #             'wind_speed_av': sum(self.speed_series) / float(len(self.speed_series)),
        #             'wind_hist': dict(wind_counter),
        #         }
        #         }
        return {'time': self.time,
                'wind_speed': self.speed_series[-1],
                }

    # increment time by one hour and return new value
    def getNext(self):
        self.randoms1.append(random.uniform(0, 1))
        self.randoms2.append(random.uniform(0, 1))
        self.state = next(j for j, v in enumerate(self.MTMcum[self.state]) if v > self.randoms1[-1])
        self.states_series.append(self.state)
        self.speed_series.append(self.WindSpeed[self.state] - 0.5 + (self.randoms2[-1] * int(self.l_categ)))
        self.time = self.time + datetime.timedelta(seconds=self.time_delta)
        return self.getCurrent()

