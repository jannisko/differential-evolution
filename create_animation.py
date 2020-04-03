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

    fig = plt.figure(figsize=(4.5, 3.375), dpi=100)
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
        plt.xlabel('x', fontsize=8)
        plt.ylabel('x²', fontsize=8)
        plt.text(5, 72,
            f'Epoch: {i:03}/{frame_num}; Best loss: {opt.get_best_point()[1]:08.5f}',
            fontsize=7
        )
        #plt.title('An example of a quadratic regression using differential evolution')
        for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(7)
        for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(7)
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


from mpl_toolkits.mplot3d import Axes3D
from differential_evolution import ackley

def animate_ackley(
    output_gif=False,
    output_mp4=True,
    frame_num=180,
    fps=30,
    frames_per_update=15,
    generations_per_update=5
):
    
    # figsize was chosen to fit github's README size
    fig = plt.figure(figsize=(4.5, 3.375), dpi=100)
    #fig = plt.figure()
    ax = fig.gca(projection='3d')
    # remove background grid, fill and axis
    ax.grid(False)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    plt.axis('off')
    # tighter fit to window
    plt.tight_layout()

    # create surface values
    x = np.arange(-5, 5, 0.05)
    y = np.arange(-5, 5, 0.05)
    xx, yy = np.meshgrid(x, y)
    z = np.empty(xx.shape, dtype=np.float)
    for i, (px, py) in enumerate(zip(xx,yy)):
        z[i] = ackley.ackley_function(px, py)

    a = tf.Variable(0.0)
    b = tf.Variable(0.0)

    # wrapper function so ackley can be called everytime a and b change
    loss = lambda : ackley.ackley_function(a,b)

    opt = DifferentialEvolution(loss, [a,b], boundaries=5)

    def animate(i):
        
        ax.view_init(elev=30.0, azim=i)

        
        if(i % frames_per_update == 0):
            for _ in range(generations_per_update):
                opt.next_generation()
            plt.cla()
            ax.plot_surface(xx, yy, z, cmap=cm.gnuplot,
                    linewidth=0, antialiased=False, alpha=0.3)
            population = opt.get_current_population()
            ax.scatter(
                population.T[0],
                population.T[1],
                ackley.ackley_function(population.T[0], population.T[1]),
                marker='o', s=40, c='#000000', alpha=1
            )
            ax.text2D(
                0.5, 0.97,
                f'Epoch: {i:03}/{frame_num}; Best loss: {opt.get_best_point()[1]:08.5f}',
                fontsize=7,
                transform=ax.transAxes
            )

        sys.stdout.write("\r%d%%" % ((i/frame_num)*100))
        sys.stdout.flush()

        ax.set_ylim((-5, 5))
        ax.set_xlim((-5, 5))
        ax.set_zlim((0, 15))

        ax.set_xlabel('x', fontsize=8)
        ax.set_ylabel('y', fontsize=8)
        ax.set_zlabel('ackley(x,y)', fontsize=8)

        for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(7)
        for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(7)
        for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(7)

        return fig,

    def init():
        return fig,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=frame_num, interval=100, blit=True)


    if(output_gif):
        anim.save('animations/ackley_optimization.gif', fps=fps, writer='imagemagick')

    if(output_mp4):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, bitrate=3000)
        anim.save('animations/ackley_optimization.mp4', writer=writer)

if __name__ == "__main__":
    animate_models(output_gif=True, output_mp4=False)
    animate_ackley(output_gif=True, output_mp4=False)