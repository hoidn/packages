import numpy as np 

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return 0.
        elif x > xs[-1]:
            return 0.
        else:
            return interpolator(x)

    def ufunclike(xs):
        try:
            iter(xs)
        except TypeError:
            xs = np.array([xs])
        return np.array(list(map(pointwise, np.array(xs))))

    return ufunclike
