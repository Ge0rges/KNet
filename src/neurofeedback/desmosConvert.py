import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

showSliders = False
# boolean that controls
# if sliders are displayed

# ===== params =====#

params = {
    'L'    : 6,
    'g'    : 1,
    'a'    : 1,
    'b'    : 3,
    'speed': 1
}

curves = None
# these are in a dict
# so they are modifiable
# in a loop

P = lambda: 2 * params['L']
o = np.pi / 180

A_1 = lambda t, s: params['a'] * np.cos(np.pi * P() * t - s) + params['b']
A_2 = lambda t, s: params['g'] * params['a'] * np.cos(np.pi * P() * t - 120 * o - s) + params['b']
A_3 = lambda t, s: params['a'] * np.cos(np.pi * P() * t - 240 * o - s) + params['b']

'''
note:
    - t is the point on the curve, within [0,1]
    - s is the rotation of all the curves about the origin

desmos original here:
https://www.desmos.com/calculator/wfb5149tff
'''

# ===== matplotlib config stuff =====#

minAxisVal = -5
maxAxisVal = 5
# variables axes length

markerSize = 7
# controls the size of the dots

plt.style.use('dark_background')
# black background

mpl.rcParams['toolbar'] = 'None'
# turn off toolbar

fig, ax = plt.subplots(1, 1)
# get figures for plotting

ax.set_xlim([minAxisVal, maxAxisVal])
ax.set_ylim([minAxisVal, maxAxisVal])
# set axis length

ax.set_facecolor((0, 0, 0))
# black plot

ax.set_aspect(aspect=1)
# set aspect ratio of the graph to 1:1

plt.axis('off')


# turn off axes
# needs to be done after calling subplot


class Curve:

    def __init__(self, func: 'func of t & s', color):
        self.func = func
        self.color = color
        self.dot, = ax.plot([], [], 'o', color=color, markersize=markerSize)
        # this is the actual dot itself
        self.path, = ax.plot([], [], color=color)
        # this is the line/guide of the dot path
        self.plotPath()

    def getX(self, t, s=0):
        return self.func(t, s) * np.cos(2 * np.pi * t)

    def getY(self, t, s=0):
        _t = params['speed'] * t
        return self.func(t, s) * np.sin(2 * np.pi * t)

    def plotPath(self):
        xs, ys = [], []
        for t in np.linspace(0, 1, 1000):
            xs.append(self.getX(t))
            ys.append(self.getY(t))

        self.path.set_data(xs, ys)

    def plotPoint(self, t):
        _t = params['speed'] * t
        # use the parameter `speed` as
        # a modifier to the scale of t

        self.dot.set_data(
            self.getX(_t),
            self.getY(_t)
        )


class Params:
    count = 0

    def __init__(self, name, valmin, valmax, valinit, valstep):
        self.name = name

        y = 0.1 + Params.count * 0.04
        Params.count += 1
        # new sliders increment a count that changes
        # the y axis of the slider so sliders
        # don't overlap on the plot.

        self.slider = mpl.widgets.Slider(
            plt.axes([0.2, y, 0.6, 0.02]),
            name + ' ',
            valmin, valmax,
            valinit=valinit,
            valstep=valstep
        )

        self.slider.on_changed(self.update)
        # calls update when the slider has been dragged

    def update(self, val):
        global params, curves

        params[self.name] = self.slider.val

        for curve in curves:
            curve.plotPath()
            # replot paths now that
            # params have changed


def main():

    if showSliders:
        # if we're plotting sliders

        plt.subplots_adjust(left=0.1, bottom=0.3)
        # shift some space for the sliders at the bottom

        paramSliders = [
            Params('speed', 0, 10, 1, 1),
            Params('b', -10, 10, 3, 1),
            Params('a', -10, 10, 1, 0.5),
            Params('g', -1, 1, 1, 0.2),
            Params('L', 1, 15, 6, 1)
        ]
    global curves

    curves = [
        Curve(A_1, 'red'),
        Curve(A_2, 'green'),
        Curve(A_3, 'blue')
    ]

    def update_dots(t):
        # note, t is going from 0->1
        for curve in curves:
            curve.plotPoint(t)

    ani = animation.FuncAnimation(
        fig,
        update_dots,
        frames=np.linspace(0, 1, 1000),
        interval=1
    )
    # this animation passes in a parameter `frames`
    # to the function update_dots, every <interval> ms

    plt.show()

if __name__ == '__main__':
    main()
