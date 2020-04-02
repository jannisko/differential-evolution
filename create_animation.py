from differential_evolution import DifferentialEvolution
from differential_evolution.losses import create_huber
from differential_evolution.models import create_quadratic
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
import sys

def animate_models(
    output_gif=False,
    output_mp4=True,
    frame_num=300,
    fps=30
):

    x = [1,2,3,4,5,6,7,8]
    y = [1,4,9,16,25,36,49,64]

    a = tf.Variable(0.0)
    b = tf.Variable(0.0)
    c = tf.Variable(0.0)

    model = create_quadratic(x, a, b, c)

    loss = create_huber(model, y)

    fig = plt.figure()
    ax = fig.gca()

    plot_model = create_quadratic(np.linspace(0,10,100), a, b, c)

    opt = DifferentialEvolution(loss, [a,b,c])

    def animate(i):
        opt.next_generation()
        plt.cla()
        for point in opt.current_population:
            a.assign(point[0][0])
            b.assign(point[1][0])
            c.assign(point[2][0])
            ax.plot(np.linspace(0,10,100), plot_model(), zorder=-1)
        ax.scatter(x,y, zorder=1, marker='^', c='000000')
        sys.stdout.write("\r%d%%" % ((i/frame_num)*100))
        sys.stdout.flush()
        ax.set_ylim((0, 70))
        ax.set_xlim((0, 10))
        plt.xlabel('x')
        plt.ylabel('x²')
        print(fig.get_size_inches()*fig.dpi)
        plt.text(5, 72, 
            f'Frame: {i:03}/{frame_num}; Best loss: {opt.get_best_point()[1]:08.5f}'
        )
        #plt.title('An example of a quadratic regression using differential evolution')
        return fig,
        

    def init():
        return fig,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=frame_num, interval=100, blit=True)


    if(output_gif):
        anim.save('animations/quadratic_models.gif', fps=fps, writer='imagemagick')

    if(output_mp4):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, bitrate=3000)
        anim.save('animations/quadratic_models.mp4', writer=writer)



if __name__ == "__main__":
    animate_models()