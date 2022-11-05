import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.cm as cm
import sys
import os
sys.path.append('../')


def visualize_process_test(pop_history: list, test_problem, problem_name: str, save_path):
    plt.ioff()
    fig, ax = plt.subplots()
    pop_plot = ax.plot([], [], 'go', markerfacecolor='none')[0]
    # pop2_plot = ax.plot([], [], 'go', markerfacecolor='none')[0]
    # archive_plot = ax.plot([], [], 'r^', markerfacecolor='none')[0]
    # pareto_plot = ax.plot([], [], 'k.')[0]
    gen_opt_plot = ax.plot([], [], 'ro', markerfacecolor='none')[0]
    pf_region_x, pf_region_y, pf_region_z = test_problem.get_pf_region()
    im = ax.imshow(pf_region_z, cmap=cm.gray, origin='lower',
                 extent=[pf_region_x[0][0], pf_region_x[0][-1], pf_region_y[0][0], pf_region_y[-1][0]], vmax=abs(pf_region_z / 9).max(),
                 vmin=-abs(pf_region_z).max())

    def update(frame):
        pop_data = frame['pop_data']
        opt_data = frame['opt']
        ax.set_xlim(pf_region_x[0][0], max(pf_region_x[0][-1] + 1, max(pop_data[:, 1]) + 0.2 * max(pop_data[:, 1])))
        ax.set_ylim(pf_region_y[0][0], max(pf_region_y[-1][0] + 1, max(pop_data[:, 1]) + 0.2 * max(pop_data[:, 1])))
        # archive_plot.set_data(archive_data[:, 0], archive_data[:, 1])
        pop_plot.set_data(pop_data[:, 0], pop_data[:, 1])
        gen_opt_plot.set_data(opt_data[:, 0], opt_data[:, 1])
        ax.set_title(f"problem: {problem_name}  gen: {frame['gen']} n_eval: {frame['n_eval']}")
        # plt.pause(0.5)

    def init():
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return pop_plot

    frame_data = []
    for i, alg in enumerate(pop_history):
        frame_data.append({'gen': i, 'pop_data': alg.pop.get('F'), 'opt': alg.opt.get('F'),
                           'n_eval': alg.evaluator.n_eval})

    ani = FuncAnimation(fig, update, frames=frame_data, init_func=init, repeat=False)
    ani.save(save_path + '/' + f"{problem_name}.gif", dpi=300, writer=PillowWriter(fps=20))


def visualize_process_test_3d(pop_history: list, test_problem, problem_name: str, save_path):
    plt.ioff()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.view_init(elev=10., azim=150)
    pop_plot = ax.plot([], [], [], 'o')[0]
    pf_region_x, pf_region_y, pf_region_z = test_problem.get_pf_region()
    surf = ax.plot_surface(pf_region_x, pf_region_y, pf_region_z, color='gray', shade=False)
    z_d_max = pf_region_z[~np.isnan(pf_region_z)]
    def update(frame):

        pop_data = frame['pop_data']
        # opt_data = frame['opt']
        ax.set_xlim(-1, max(pf_region_x[0][-1] + 1, max(pop_data[:, 0]) + 0.2 * max(pop_data[:, 0])))
        ax.set_ylim(-1, max(pf_region_y[-1][0] + 1, max(pop_data[:, 1]) + 0.2 * max(pop_data[:, 1])))
        ax.set_zlim(min(pop_data[:, 2] - 1), max(max(z_d_max), max(pop_data[:, 2]) + 0.2 * max(pop_data[:, 2])))
        # archive_plot.set_data(archive_data[:, 0], archive_data[:, 1])
        pop_plot.set_data(pop_data[:, 0], pop_data[:, 1])
        pop_plot.set_3d_properties(pop_data[:, 2])
        # gen_opt_plot.set_data(opt_data[:, 0], opt_data[:, 1], opt_data[:, 2])
        ax.set_title(f"problem: {problem_name}  gen: {frame['gen']} n_eval: {frame['n_eval']}")
        # plt.pause(0.5)

    def init():
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        return pop_plot

    frame_data = []
    for i, alg in enumerate(pop_history):
        frame_data.append({'gen': i, 'pop_data': alg.pop.get('F'), 'opt': alg.opt.get('F'),
                           'n_eval': alg.evaluator.n_eval})

    ani = FuncAnimation(fig, update, frames=frame_data, init_func=init, repeat=False)
    ani.save(save_path + '/' + f"{problem_name}.gif", dpi=300, writer=PillowWriter(fps=20))


def visual_process_dual_pop(pop_history: list, test_problem, problem_name: str, save_path):
    plt.ioff()
    fig, ax = plt.subplots()
    pop_plot = ax.plot([], [], 'go', markerfacecolor='none')[0]
    pop2_plot = ax.plot([], [], 'bo', markerfacecolor='none')[0]
    # archive_plot = ax.plot([], [], 'r^', markerfacecolor='none')[0]
    pareto_plot = ax.plot([], [], 'k.')[0]
    gen_opt_plot = ax.plot([], [], 'ro', markerfacecolor='none')[0]
    pf_region_x, pf_region_y, pf_region_z = test_problem.get_pf_region()
    im = ax.imshow(pf_region_z, cmap=cm.gray, origin='lower',
                 extent=[pf_region_x[0][0], pf_region_x[0][-1], pf_region_y[0][0], pf_region_y[-1][0]], vmax=abs(pf_region_z / 9).max(),
                 vmin=-abs(pf_region_z).max())

    def update(frame):
        pop_data = frame['pop_data']
        opt_data = frame['opt']
        pop_h_data = frame['pop_h_data']
        ax.set_xlim(pf_region_x[0][0], max(pf_region_x[0][-1] + 1, max(pop_data[:, 1]) + 0.2 * max(pop_data[:, 1])))
        ax.set_ylim(pf_region_y[0][0], max(pf_region_y[-1][0] + 1, max(pop_data[:, 1]) + 0.2 * max(pop_data[:, 1])))

        pop_plot.set_data(pop_data[:, 0], pop_data[:, 1])
        pop2_plot.set_data(pop_h_data[:, 0], pop_h_data[:, 1])
        gen_opt_plot.set_data(opt_data[:, 0], opt_data[:, 1])
        ax.set_title(f"problem: {problem_name}  gen: {frame['gen']} n_eval: {frame['n_eval']}")


    def init():
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return pop_plot

    frame_data = []
    for i, alg in enumerate(pop_history):
        frame_data.append({'gen': i, 'pop_data': alg.pop.get('F'), 'pop_h_data': alg.pop_h.get('F'),
                           'opt': alg.opt.get('F'),
                           'n_eval': alg.evaluator.n_eval})

    ani = FuncAnimation(fig, update, frames=frame_data, init_func=init, repeat=False)
    ani.save(save_path + '/' + f"{problem_name}.gif", dpi=300, writer=PillowWriter(fps=20))


def display_result(test_problem, pop):

    fig, ax = plt.subplots()

    pf_region_x, pf_region_y, pf_region_z = test_problem.get_pf_region()
    im = ax.imshow(pf_region_z, cmap=cm.gray, origin='lower',
                 extent=[pf_region_x[0][0], pf_region_x[0][-1], pf_region_y[0][0], pf_region_y[-1][0]], vmax=abs(pf_region_z / 9).max(),
                 vmin=-abs(pf_region_z).max())

    ax.plot(pop[:, 0], pop[:, 1], 'go', markerfacecolor='none')
    ax.set_xlim(pf_region_x[0][0], max(pf_region_x[0][-1] + 1, max(pop[:, 1]) + 0.2 * max(pop[:, 1])))
    ax.set_ylim(pf_region_y[0][0], max(pf_region_y[-1][0] + 1, max(pop[:, 1]) + 0.2 * max(pop[:, 1])))
    plt.show()


def display_result_fix_lim(test_problem, res, save_path, problem_name):

    fig, ax = plt.subplots()
    # fig.set_size_inches(10.5, 10.5, forward=True)
    # fig.set_dpi(100)
    # ax.axis('equal')
    # plt.axis('scaled')
    pf_region_x, pf_region_y, pf_region_z = test_problem.get_pf_region()
    pf = test_problem.pareto_front()
    pop = res.pop.get('F')
    pop_h = res.algorithm.pop_h.get('F')
    optimal = res.opt.get('F')
    a, b, c, d, = pf_region_x[0][0], pf_region_x[0][-1], pf_region_y[0][0], pf_region_y[-1][0]
    im = ax.imshow(pf_region_z, cmap=cm.gray, origin='lower',
                 extent=[pf_region_x[0][0], pf_region_x[0][-1], pf_region_y[0][0], pf_region_y[-1][0]], vmax=abs(pf_region_z / 9).max(),
                 vmin=-abs(pf_region_z).max(), aspect='auto')

    ax.plot(pop[:, 0], pop[:, 1], 'go', markerfacecolor='none', label='pop_o')
    ax.plot(pop_h[:, 0], pop_h[:, 1], 'bo', markerfacecolor='none', label='pop_h')
    ax.plot(optimal[:, 0], optimal[:, 1], 'ro', label='optimal')
    ax.plot(pf[:, 0], pf[:, 1], 'k.', label='pf')

    ax.set_xlim(pf_region_x[0][0], pf_region_x[0][-1])
    ax.set_ylim(pf_region_y[0][0], pf_region_y[-1][0])
    ax.legend(loc='lower left')
    fig.savefig(os.path.join(save_path, problem_name+'_last_fix.png'), dpi=300)
    plt.show()


def visual_process_dual_pop_fix(pop_history: list, test_problem, problem_name: str, save_path):
    plt.ioff()
    fig, ax = plt.subplots()
    pop_plot = ax.plot([], [], 'go', markerfacecolor='none', label='pop_o')[0]
    pop2_plot = ax.plot([], [], 'bo', markerfacecolor='none', label='pop_h')[0]
    # archive_plot = ax.plot([], [], 'r^', markerfacecolor='none')[0]
    pareto_plot = ax.plot([], [], 'k.', label='pf')[0]
    gen_opt_plot = ax.plot([], [], 'ro', markerfacecolor='none', label='optimal')[0]
    pf_region_x, pf_region_y, pf_region_z = test_problem.get_pf_region()
    im = ax.imshow(pf_region_z, cmap=cm.gray, origin='lower',
                 extent=[pf_region_x[0][0], pf_region_x[0][-1], pf_region_y[0][0], pf_region_y[-1][0]], vmax=abs(pf_region_z / 9).max(),
                 vmin=-abs(pf_region_z).max(), aspect='auto')
    ax.legend(loc='lower left')

    def update(frame):
        pop_data = frame['pop_data']
        opt_data = frame['opt']
        pop_h_data = frame['pop_h_data']
        pf_data = frame['pf']
        ax.set_xlim(pf_region_x[0][0], pf_region_x[0][-1])
        ax.set_ylim(pf_region_y[0][0], pf_region_y[-1][0])
        pareto_plot.set_data(pf_data[:, 0], pf_data[:, 1])
        pop_plot.set_data(pop_data[:, 0], pop_data[:, 1])
        pop2_plot.set_data(pop_h_data[:, 0], pop_h_data[:, 1])
        gen_opt_plot.set_data(opt_data[:, 0], opt_data[:, 1])
        # ax.legend()
        ax.set_title(f"problem: {problem_name}  gen: {frame['gen']} n_eval: {frame['n_eval']}")


    def init():
        ax.set_xlim(pf_region_x[0][0], pf_region_x[0][-1])
        ax.set_ylim(pf_region_y[0][0], pf_region_y[-1][0])
        return pop_plot

    frame_data = []
    for i, alg in enumerate(pop_history):
        frame_data.append({'gen': i, 'pop_data': alg.pop.get('F'), 'pop_h_data': alg.pop_h.get('F'),
                           'opt': alg.opt.get('F'), 'pf': alg.problem.pareto_front(),
                           'n_eval': alg.evaluator.n_eval})

    ani = FuncAnimation(fig, update, frames=frame_data, init_func=init, repeat=False)
    ani.save(save_path + '/' + f"{problem_name}_fix.gif", dpi=300, writer=PillowWriter(fps=20))


def visualize_process_one_pop_fix(pop_history: list, test_problem, problem_name: str, save_path):
    plt.ioff()
    fig, ax = plt.subplots()
    pop_plot = ax.plot([], [], 'go', markerfacecolor='none')[0]
    gen_opt_plot = ax.plot([], [], 'ro', markerfacecolor='none')[0]
    pf_region_x, pf_region_y, pf_region_z = test_problem.get_pf_region()
    im = ax.imshow(pf_region_z, cmap=cm.gray, origin='lower',
                 extent=[pf_region_x[0][0], pf_region_x[0][-1], pf_region_y[0][0], pf_region_y[-1][0]], vmax=abs(pf_region_z / 9).max(),
                 vmin=-abs(pf_region_z).max(), aspect='auto')

    def update(frame):
        pop_data = frame['pop_data']
        opt_data = frame['opt']
        ax.set_xlim(pf_region_x[0][0], pf_region_x[0][-1])
        ax.set_ylim(pf_region_y[0][0], pf_region_y[-1][0])
        # archive_plot.set_data(archive_data[:, 0], archive_data[:, 1])
        pop_plot.set_data(pop_data[:, 0], pop_data[:, 1])
        gen_opt_plot.set_data(opt_data[:, 0], opt_data[:, 1])
        ax.set_title(f"problem: {problem_name}  gen: {frame['gen']} n_eval: {frame['n_eval']}")
        # plt.pause(0.5)

    def init():
        ax.set_xlim(pf_region_x[0][0], pf_region_x[0][-1])
        ax.set_ylim(pf_region_y[0][0], pf_region_y[-1][0])
        return pop_plot

    frame_data = []
    for i, alg in enumerate(pop_history):
        frame_data.append({'gen': i, 'pop_data': alg.pop.get('F'), 'opt': alg.opt.get('F'),
                           'n_eval': alg.evaluator.n_eval})

    ani = FuncAnimation(fig, update, frames=frame_data, init_func=init, repeat=False)
    ani.save(save_path + '/' + f"{problem_name}.gif", dpi=300, writer=PillowWriter(fps=20))


if __name__ == '__main__':
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.problems import get_problem
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.optimize import minimize
    from DisplayProblem.displaymw import *


    problem = DisplayMW4()
    algorithm = NSGA2(pop_size=100)
    res = minimize(problem, algorithm, ("n_gen", 100), verbose=True, save_history=True)

    his = res.history
    visualize_process_test_3d(his, problem, "mw14", "G:/code/MyProject/CSCMO/visual_result")

