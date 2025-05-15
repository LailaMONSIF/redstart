import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Redstart: A Lightweight Reusable Booster""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="public/images/redstart.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Project Redstart is an attempt to design the control systems of a reusable booster during landing.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In principle, it is similar to SpaceX's Falcon Heavy Booster.

    >The Falcon Heavy booster is the first stage of SpaceX's powerful Falcon Heavy rocket, which consists of three modified Falcon 9 boosters strapped together. These boosters provide the massive thrust needed to lift heavy payloads—like satellites or spacecraft—into orbit. After launch, the two side boosters separate and land back on Earth for reuse, while the center booster either lands on a droneship or is discarded in high-energy missions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(
        mo.Html("""
    <iframe width="560" height="315" src="https://www.youtube.com/embed/RYUr-5PYA7s?si=EXPnjNVnqmJSsIjc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>""")
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell(hide_code=True)
def _():
    import scipy
    import scipy.integrate as sci

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    return FFMpegWriter, FuncAnimation, mpl, np, plt, scipy, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The Model

    The Redstart booster in model as a rigid tube of length $2 \ell$ and negligible diameter whose mass $M$ is uniformly spread along its length. It may be located in 2D space by the coordinates $(x, y)$ of its center of mass and the angle $\theta$ it makes with respect to the vertical (with the convention that $\theta > 0$ for a left tilt, i.e. the angle is measured counterclockwise)

    This booster has an orientable reactor at its base ; the force that it generates is of amplitude $f>0$ and the angle of the force with respect to the booster axis is $\phi$ (with a counterclockwise convention).

    We assume that the booster is subject to gravity, the reactor force and that the friction of the air is negligible.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image(src="public/images/geometry.svg"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Constants

    For the sake of simplicity (this is merely a toy model!) in the sequel we assume that: 

      - the total length $2 \ell$ of the booster is 2 meters,
      - its mass $M$ is 1 kg,
      - the gravity constant $g$ is 1 m/s^2.

    This set of values is not realistic, but will simplify our computations and do not impact the structure of the booster dynamics.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Helpers

    ### Rotation matrix

    $$ 
    \begin{bmatrix}
    \cos \alpha & - \sin \alpha \\
    \sin \alpha &  \cos \alpha  \\
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return (R,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Videos

    It will be very handy to make small videos to visualize the evolution of our booster!
    Here is an example of how such videos can be made with Matplotlib and displayed in marimo.
    """
    )
    return


@app.cell(hide_code=True)
def _(FFMpegWriter, FuncAnimation, mo, np, plt, tqdm):
    def make_video(output):
        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        num_frames = 100
        fps = 30 # Number of frames per second

        def animate(frame_index):    
            # Clear the canvas and redraw everything at each step
            plt.clf()
            plt.xlim(0, 2*np.pi)
            plt.ylim(-1.5, 1.5)
            plt.title(f"Sine Wave Animation - Frame {frame_index+1}/{num_frames}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)

            x = np.linspace(0, 2*np.pi, 100)
            phase = frame_index / 10
            y = np.sin(x + phase)
            plt.plot(x, y, "r-", lw=2, label=f"sin(x + {phase:.1f})")
            plt.legend()

            pbar.update(1)

        pbar = tqdm(total=num_frames, desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=num_frames)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")

    _filename = "wave_animation.mp4"
    make_video(_filename)
    mo.show_code(mo.video(src=_filename))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Getting Started""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Constants

    Define the Python constants `g`, `M` and `l` that correspond to the gravity constant, the mass and half-length of the booster.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    g = 1.0
    M = 1.0
    l = 1
    return M, g, l


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Forces

    Compute the force $(f_x, f_y) \in \mathbb{R}^2$ applied to the booster by the reactor.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    f_x & = -f \sin (\theta + \phi) \\
    f_y & = +f \cos(\theta +\phi)
    \end{align*}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Center of Mass

    Give the ordinary differential equation that governs $(x, y)$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    M \ddot{x} & = -f \sin (\theta + \phi) \\
    M \ddot{y} & = +f \cos(\theta +\phi) - Mg
    \end{align*}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Moment of inertia

    Compute the moment of inertia $J$ of the booster and define the corresponding Python variable `J`.
    """
    )
    return


@app.cell
def _(M, l):
    J = M * l * l / 3
    J
    return (J,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Tilt

    Give the ordinary differential equation that governs the tilt angle $\theta$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    J \ddot{\theta} = - \ell (\sin \phi)  f
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Simulation

    Define a function `redstart_solve` that, given as input parameters: 

      - `t_span`: a pair of initial time `t_0` and final time `t_f`,
      - `y0`: the value of the state `[x, dx, y, dy, theta, dtheta]` at `t_0`,
      - `f_phi`: a function that given the current time `t` and current state value `y`
         returns the values of the inputs `f` and `phi` in an array.

    returns:

      - `sol`: a function that given a time `t` returns the value of the state `[x, dx, y, dy, theta, dtheta]` at time `t` (and that also accepts 1d-arrays of times for multiple state evaluations).

    A typical usage would be:

    ```python
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    ```

    Test this typical example with your function `redstart_solve` and check that its graphical output makes sense.
    """
    )
    return


@app.cell(hide_code=True)
def _(J, M, g, l, np, scipy):
    def redstart_solve(t_span, y0, f_phi):
        def fun(t, state):
            x, dx, y, dy, theta, dtheta = state
            f, phi = f_phi(t, state)
            d2x = (-f * np.sin(theta + phi)) / M
            d2y = (+ f * np.cos(theta + phi)) / M - g
            d2theta = (- l * np.sin(phi)) * f / J
            return np.array([dx, d2x, dy, d2y, dtheta, d2theta])
        r = scipy.integrate.solve_ivp(fun, t_span, y0, dense_output=True)
        return r.sol
    return (redstart_solve,)


@app.cell(hide_code=True)
def _(l, np, plt, redstart_solve):
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0) = - 2*\ell$,  can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    % y(t)
    y(t)
    = \frac{2(5-\ell)}{125}\,t^3
      + \frac{3\ell-10}{25}\,t^2
      - 2\,t
      + 10
    $$

    $$
    % f(t)
    f(t)
    = M\!\Bigl[
        \frac{12(5-\ell)}{125}\,t
        + \frac{6\ell-20}{25}
        + g
      \Bigr].
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(M, g, l, np, plt, redstart_solve):

    def smooth_landing_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example()
    return


@app.cell
def _(M, g, l, np, plt, redstart_solve):
    def smooth_landing_example_force():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example_force()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Drawing

    Create a function that draws the body of the booster, the flame of its reactor as well as its target landing zone on the ground (of coordinates $(0, 0)$).

    The drawing can be very simple (a rectangle for the body and another one of a different color for the flame will do perfectly!).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image("public/images/booster_drawing.png"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Make sure that the orientation of the flame is correct and that its length is proportional to the force $f$ with the length equal to $\ell$ when $f=Mg$.

    The function shall accept the parameters `x`, `y`, `theta`, `f` and `phi`.
    """
    )
    return


@app.cell(hide_code=True)
def _(M, R, g, l, mo, mpl, np, plt):
    def draw_booster(x=0, y=l, theta=0.0, f=0.0, phi=0.0, axes=None, **options):
        L = 2 * l
        if axes is None:
            _fig, axes = plt.subplots()

        axes.set_facecolor('#F0F9FF') 

        ground = np.array([[-2*l, 0], [2*l, 0], [2*l, -l], [-2*l, -l], [-2*l, 0]]).T
        axes.fill(ground[0], ground[1], color="#E3A857", **options)

        b = np.array([
            [l/10, -l], 
            [l/10, l], 
            [0, l+l/10], 
            [-l/10, l], 
            [-l/10, -l], 
            [l/10, -l]
        ]).T
        b = R(theta) @ b
        axes.fill(b[0]+x, b[1]+y, color="black", **options)

        ratio = l / (M*g) # when f= +MG, the flame length is l 

        flame = np.array([
            [l/10, 0], 
            [l/10, - ratio * f], 
            [-l/10, - ratio * f], 
            [-l/10, 0], 
            [l/10, 0]
        ]).T
        flame = R(theta+phi) @ flame
        axes.fill(
            flame[0] + x + l * np.sin(theta), 
            flame[1] + y - l * np.cos(theta), 
            color="#FF4500", 
            **options
        )

        return axes

    _axes = draw_booster(x=0.0, y=20*l, theta=np.pi/8, f=M*g, phi=np.pi/8)
    _fig = _axes.figure
    _axes.set_xlim(-4*l, 4*l)
    _axes.set_ylim(-2*l, 24*l)
    _axes.set_aspect("equal")
    _axes.grid(True)
    _MaxNLocator = mpl.ticker.MaxNLocator
    _axes.xaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.yaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.set_axisbelow(True)
    mo.center(_fig)
    return (draw_booster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualisation

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin the with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


@app.cell(hide_code=True)
def _(draw_booster, l, mo, np, plt, redstart_solve):
    def sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_1()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_2():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_2()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_3()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_4():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_4()
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    draw_booster,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_1.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_1())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_2():
        L = 2*l

        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_2.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_2())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_3.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_3())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_4():
        L = 2*l
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_4.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_4())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Linearized Dynamics""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Equilibria

    We assume that $|\theta| < \pi/2$, $|\phi| < \pi/2$ and that $f > 0$. What are the possible equilibria of the system for constant inputs $f$ and $\phi$ and what are the corresponding values of these inputs?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Nous cherchons les *états d’équilibre* du système, c’est-à-dire les configurations où :

    $$
    \dot{x} = \dot{y} = \dot{\theta} = 0, \quad \ddot{x} = \ddot{y} = \ddot{\theta} = 0
    $$

    D’après les équations fournies, la dynamique du système est :
    ### Équations du mouvement

    Le système dynamique du booster est donné par les équations suivantes :

    #### Translation horizontale :

    $$
    M \ddot{x} = -f \sin(\theta + \phi)
    $$

    #### Translation verticale :

    $$
    M \ddot{y} = f \cos(\theta + \phi) - Mg
    $$

    #### Rotation (inclinaison) :

    $$
    J \ddot{\theta} = -\ell f \sin(\phi)
    $$

    ---

    ### Conditions à l'équilibre

    #### 1. Équilibre horizontal

    $$
    \ddot{x} = 0 \Rightarrow \sin(\theta + \phi) = 0 \Rightarrow \theta + \phi = k\pi
    $$

    Avec les contraintes \( |\theta| < \frac{\pi}{2} \) et \( |\phi| < \frac{\pi}{2} \), la seule solution acceptable est :

    $$
    \boxed{\theta + \phi = 0}
    \quad \Rightarrow \boxed{\phi = -\theta}
    $$

    ---

    #### 2. Équilibre vertical

    $$
    \ddot{y} = 0 \Rightarrow f \cos(\theta + \phi) = Mg
    $$

    En remplaçant \( \theta + \phi = 0 \) :

    $$
    \cos(0) = 1 \Rightarrow \boxed{f = Mg}
    $$

    ---

    #### 3. Équilibre de rotation

    $$
    \ddot{\theta} = 0 \Rightarrow \sin(\phi) = 0 \Rightarrow \boxed{\phi = 0}
    \quad \Rightarrow \boxed{\theta = 0}
    $$

    ---



    Le système possède un *équilibre unique* (sous les hypothèses \( f > 0 \), \( |\theta| < \frac{\pi}{2} \), \( |\phi| < \frac{\pi}{2} \)) :

    $$
    \boxed{
    \theta = 0, \quad \phi = 0, \quad f = Mg
    }
    $$

    Dans cet état :
    - Le booster est *parfaitement vertical*.
    - La poussée est *verticale vers le haut*.
    - Elle compense exactement la gravité.
    - Il n’y a *ni mouvement ni rotation*.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linearized Model

    Introduce the error variables $\Delta x$, $\Delta y$, $\Delta \theta$, and $\Delta f$ and $\Delta \phi$ of the state and input values with respect to the generic equilibrium configuration.
    What are the linear ordinary differential equations that govern (approximately) these variables in a neighbourhood of the equilibrium?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We define error variables around the equilibrium point \( \theta = 0, \phi = 0, f = Mg \):

    - \( \Delta x = x - x_e \)
    - \( \Delta y = y - y_e \)
    - \( \Delta \theta = \theta \)
    - \( \Delta f = f - Mg \)
    - \( \Delta \phi = \phi \)

    ### Linearized Equations of Motion:

    \[
    \begin{aligned}
    \ddot{\Delta x} &= -g (\Delta \theta + \Delta \phi) \\
    \ddot{\Delta y} &= \frac{1}{M} \Delta f \\
    \ddot{\Delta \theta} &= -\frac{\ell g}{J} \Delta \phi
    \end{aligned}
    \]
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Standard Form

    What are the matrices $A$ and $B$ associated to this linear model in standard form?
    Define the corresponding NumPy arrays `A` and `B`.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    We express the linearized system dynamics in the standard state-space form:

    \[
    \dot{\mathbf{x}} = A \mathbf{x} + B \mathbf{u}
    \]

    where

    - \(\mathbf{x} = [\Delta x, \Delta y, \Delta \theta, \Delta \dot{x}, \Delta \dot{y}, \Delta \dot{\theta}]^T\) is the state error vector around the equilibrium,
    - \(\mathbf{u} = [\Delta f, \Delta \phi]^T\) is the input error vector.

    The matrices \(A\) and \(B\) capture the system dynamics and how inputs affect accelerations respectively.

    ---

    - Gravity \(g\), mass \(M\), moment of inertia \(J\), and lever arm \(l\) are constants.
    - The linearized equations show coupling between orientation and accelerations.
    - The resulting matrices \(A\) and \(B\) are:

    \[
    A = 
    \begin{bmatrix}
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & -g & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    ,
    \quad
    B = 
    \begin{bmatrix}
    0 & 0 \\
    0 & 0 \\
    0 & 0 \\
    0 & -g \\
    1/M & 0 \\
    0 & -\frac{l g}{J}
    \end{bmatrix}
    \]
    """
    )
    return


@app.cell
def _(J, M, g, l, np):
    A = np.array([
        [0, 0, 0, 1, 0, 0],                # d(Δx)/dt = Δvx
        [0, 0, 0, 0, 1, 0],                # d(Δy)/dt = Δvy
        [0, 0, 0, 0, 0, 1],                # d(Δθ)/dt = Δω
        [0, 0, -g, 0, 0, 0],               # d(Δvx)/dt = -g Δθ
        [0, 0, 0, 0, 0, 0],                # d(Δvy)/dt = (1/M) Δf (input)
        [0, 0, 0, 0, 0, 0]                 # d(Δω)/dt = (-l g / J) Δφ (input)
    ])

    B = np.array([
        [0, 0],                            # Δf, Δφ don't affect position directly
        [0, 0],
        [0, 0],
        [0, -g],                           # Δφ affects Δvx through -g
        [1/M, 0],                          # Δf affects Δvy
        [0, -l * g / J]                    # Δφ affects angular accel
    ])
    return A, B


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Stability

    Is the generic equilibrium asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Nous analysons la stabilité de l’équilibre générique à partir de la matrice linéarisée \( A \). Celle-ci contient plusieurs lignes de zéros et aucun terme d’amortissement ou de dissipation. En particulier :

    - Il existe *au moins une valeur propre nulle*, liée aux équations de translation verticale et de rotation.
    - Il n’y a *aucune valeur propre avec partie réelle strictement négative*.

    ---



    L’équilibre *n’est pas asymptotiquement stable*.

    Il est *au mieux marginalement stable*, ce qui signifie que :

    - Une petite perturbation ne s’éteindra pas d’elle-même.
    - Le système *ne revient pas naturellement à l’équilibre sans correction active*.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controllability

    Is the linearized model controllable?
    """
    )
    return


@app.cell(hide_code=True)
def _(A, B, np):
    from numpy.linalg import matrix_rank

    # Construct the controllability matrix
    n = A.shape[0]
    controllability_matrix = B
    for i in range(1, n):
        controllability_matrix = np.hstack((controllability_matrix, np.linalg.matrix_power(A, i) @ B))

    # Check rank
    rank = matrix_rank(controllability_matrix)

    print(f"Rank of controllability matrix: {rank}")
    print(f"System is controllable: {rank == n}")
    return (matrix_rank,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Lateral Dynamics

    We limit our interest in the lateral position $x$, the tilt $\theta$ and their derivatives (we are for the moment fine with letting $y$ and $\dot{y}$ be uncontrolled). We also set $f = M g$ and control the system only with $\phi$.

    What are the new (reduced) matrices $A$ and $B$ for this reduced system?
    Check the controllability of this new system.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Focusing on the lateral position \(x\) and tilt \(\theta\) with their velocities, and controlling only the thrust angle \(\phi\) (with fixed thrust \(f = Mg\)), the reduced state vector is:

    \[
    \mathbf{x}_{red} = [\Delta x, \Delta \dot{x}, \Delta \theta, \Delta \dot{\theta}]^T
    \]

    and input:

    \[
    u = \Delta \phi
    \]

    The reduced system matrices \(A_{red}\) and \(B_{red}\) are extracted from the full matrices by selecting the relevant states and input.

    We then check the controllability of this reduced system using the controllability matrix:

    \[
    \mathcal{C}{red} = \begin{bmatrix} B{red} & A_{red} B_{red} & A_{red}^2 B_{red} & \cdots \end{bmatrix}
    \]

    If \(\operatorname{rank}(\mathcal{C}_{red}) = 4\) (the dimension of the reduced state), the system remains controllable under this reduced setting.
    """
    )
    return


@app.cell(hide_code=True)
def _(A, B, matrix_rank, np):
    # Indices of states in reduced system
    indices_states = [0, 3, 2, 5]
    # Index of input phi (second input)
    index_phi = 1

    # Extract reduced A matrix
    A_red = A[np.ix_(indices_states, indices_states)]

    # Extract reduced B matrix (only column for phi)
    B_red = B[indices_states, index_phi].reshape(-1, 1)

    print("Reduced A matrix:")
    print(A_red)

    print("\nReduced B matrix:")
    print(B_red)

    # Check controllability of reduced system
    n_red = A_red.shape[0]
    controllability_matrix_red = B_red
    for k in range(1, n_red):
        controllability_matrix_red = np.hstack((controllability_matrix_red, np.linalg.matrix_power(A_red, k) @ B_red))

    rank_red = matrix_rank(controllability_matrix_red)

    print(f"\nRank of reduced controllability matrix: {rank_red}")
    print(f"Reduced system controllable: {rank_red == n_red}")
    return A_red, B_red


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linear Model in Free Fall

    Make graphs of $y(t)$ and $\theta(t)$ for the linearized model when $\phi(t)=0$,
    $x(0)=0$, $\dot{x}(0)=0$, $\theta(0) = 45 / 180  \times \pi$  and $\dot{\theta}(0) =0$. What do you see? How do you explain it?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Linear Model in Free Fall

    On observe ici le comportement latéral du booster sans aucune commande.

    ### Hypothèses :
    - $\phi(t) = 0$
    - $x(0) = 0$, $\dot{x}(0) = 0$
    - $\theta(0) = \pi/4$, $\dot{\theta}(0) = 0$

    ### Interprétation :
    L’inclinaison reste constante, donc le booster continue sa chute en biais.  
    La position latérale $x(t)$ dérive petit à petit.

    Sans correction active, le système ne revient pas à l’équilibre. Il n’est donc pas stable par lui-même.

    """
    )
    return


@app.cell(hide_code=True)
def _():
    def _():
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.integrate import solve_ivp

        # Matrice A_red : dynamique réduite (x, dx, theta, dtheta)
        A_red = np.array([
            [0, 1, 0, 0],
            [0, 0, -1.0, 0],  # influence de θ sur l'accélération de x
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])

        # Pas de commande : u = 0
        def model(t, X):
            return A_red @ X

        # Conditions initiales : x = 0, dx = 0, theta = 45°, dtheta = 0
        X0 = [0.0, 0.0, np.pi/4, 0.0]
        t_span = (0, 5)
        t_eval = np.linspace(*t_span, 1000)
        sol = solve_ivp(model, t_span, X0, t_eval=t_eval)

        # Extraire les résultats
        x_t = sol.y[0]
        theta_t = sol.y[2]

        # Tracer les courbes
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(sol.t, x_t, label="x(t)", color="blue")
        plt.xlabel("Temps (s)")
        plt.ylabel("Position latérale x")
        plt.title("Évolution de x(t) (sans commande)")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(sol.t, theta_t, label="θ(t)", color="orange")
        plt.xlabel("Temps (s)")
        plt.ylabel("Inclinaison θ (rad)")
        plt.title("Évolution de θ(t) (sans commande)")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        return plt.show()


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Manually Tuned Controller

    Try to find the two missing coefficients of the matrix 

    $$
    K =
    \begin{bmatrix}
    0 & 0 & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    manages  when
    $\Delta x(0)=0$, $\Delta \dot{x}(0)=0$, $\Delta \theta(0) = 45 / 180  \times \pi$  and $\Delta \dot{\theta}(0) =0$ to: 

      - make $\Delta \theta(t) \to 0$ in approximately $20$ sec (or less),
      - $|\Delta \theta(t)| < \pi/2$ and $|\Delta \phi(t)| < \pi/2$ at all times,
      - (but we don't care about a possible drift of $\Delta x(t)$).

    Explain your thought process, show your iterations!

    Is your closed-loop model asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Thought Process

    Since k₁ = k₂ = 0, only the angular subsystem matters:

    \[
    \dot{x}\theta = A\theta x_\theta + B_\theta u = (A_\theta - B_\theta K_\theta) x_\theta
    \]

    where:

    - \( x_\theta = \begin{bmatrix} \Delta \theta \\ \Delta \dot{\theta} \end{bmatrix} \)
  
    - \( K_\theta = \begin{bmatrix} k_3 & k_4 \end{bmatrix} \)

    We designed a simulation that:

    - Iterates over a grid of k₃ and k₄ values
    - Checks if the eigenvalues of the closed-loop system have negative real parts
    - Simulates the response using solve_ivp and checks if:
    - Δθ(t) converges to zero in ≤ 30s
    - |Δθ(t)| and |Δϕ(t)| stay below π/2 throughout

    ---

    ### Suitable Gains Found
    k3 = -1.655
    k4 = -10.000

    $$
    K =
    \begin{bmatrix}
    0 & 0 & -1.655 & -10
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$

    """
    )
    return


@app.cell(hide_code=True)
def _(A_red, B_red, np, plt):
    from scipy.linalg import eig
    from scipy.integrate import solve_ivp

    A_theta = A_red[2:4, 2:4]
    B_theta = B_red[2:4, :]

    x0_theta = np.array([np.pi/4, 0])

    def closed_loop_dynamics_manual(t, x, k3, k4):
        K_theta = np.array([k3, k4])
        u = -K_theta @ x
        dxdt = A_theta @ x + B_theta.flatten() * u
        return dxdt

    def simulate_response_manual(k3, k4, t_end=40, dt=0.01):
        t_span = (0, t_end)
        t_eval = np.arange(0, t_end, dt)
        sol = solve_ivp(closed_loop_dynamics_manual, t_span, x0_theta, args=(k3, k4), t_eval=t_eval)
        theta = sol.y[0]
        theta_dot = sol.y[1]
        u = - (k3 * theta + k4 * theta_dot)
        return sol.t, theta, u

    k3_vals = np.linspace(-10, 1, 30)
    k4_vals = np.linspace(-10, 1, 30)

    best_k3, best_k4 = None, None
    for k3_i in k3_vals:
        for k4_i in k4_vals:
            A_cl_manual = A_theta - B_theta @ np.array([[k3_i, k4_i]])
            eigvals_manual = eig(A_cl_manual)[0]
            if np.all(np.real(eigvals_manual) < 0):  # stable
                t_sim, theta_sim, u_sim = simulate_response_manual(k3_i, k4_i)
                if (np.max(np.abs(theta_sim)) < np.pi/2) and (np.max(np.abs(u_sim)) < np.pi/2):
                    settling_indices = np.where(np.abs(theta_sim) < 0.05)[0]
                    if len(settling_indices) > 0 and t_sim[settling_indices[0]] <= 20:
                        best_k3, best_k4 = k3_i, k4_i
                        break
        if best_k3 is not None:
            break

    if best_k3 is not None:
        print(f"Suitable gains found: k3 = {best_k3:.3f}, k4 = {best_k4:.3f}")
        t_sim, theta_sim, u_sim = simulate_response_manual(best_k3, best_k4)

        plt.plot(t_sim, theta_sim, label=r'$\Delta \theta(t)$')
        plt.plot(t_sim, u_sim, label=r'$\Delta \phi(t)$')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (rad)')
        plt.title('Closed-loop response with manually tuned gains')

        plt.axhline(y=np.pi/2, color='red', linestyle='--', label=r'$+\pi/2$ limit')
        plt.axhline(y=-np.pi/2, color='red', linestyle='--', label=r'$-\pi/2$ limit')
        plt.axvline(x=20, color='purple', linestyle='--', label='Settling time limit (20s)')

        plt.legend()
        plt.grid(True)

        plt.xlim(0, t_sim[-1])

        plt.show()
    else:
        print("No suitable gains found in search range.")
    return (settling_indices,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Pole Assignment

    Using pole assignement, find a matrix

    $$
    K_{pp} =
    \begin{bmatrix}
    ? & ? & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K_{pp} \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    satisfies the conditions defined for the manually tuned controller and additionally:

      - result in an asymptotically stable closed-loop dynamics,

      - make $\Delta x(t) \to 0$ in approximately $20$ sec (or less).

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To design the feedback gain matrix \( K_{pp} \) for the lateral dynamics, we use pole assignment on the reduced closed-loop system:

    \[
    A_{cl} = A_{red} - B_{red} K_{pp}
    \]

    We initially aimed for poles that would ensure *asymptotic stability* and convergence of the lateral position error \(\Delta x(t)\) within approximately 20 seconds. However, after testing different values, we found that choosing faster poles such as:

    \[
    \texttt{desired\_poles} = [-2, -2.5, -3, -3.5]
    \]

    led to a significantly improved transient response. Using the place_poles method, we computed the gain matrix \( K_{pp} \) to place the poles at these locations.

    The resulting control law:

    \[
    \Delta \phi(t) = -K_{pp} \mathbf{x}_{red}(t)
    \]

    produced a closed-loop response where \(\Delta x(t)\) starts at 1 m, briefly overshoots (up to ~1.2 m at 1 s), and then converges to zero within approximately 4 seconds — much faster than the original target, while maintaining stability and good performance.
    """
    )
    return


@app.cell(hide_code=True)
def _(A_red, B_red):
    from scipy.signal import place_poles

    desired_poles = [-1, -1.2, -1.5, -1.8]

    place_obj = place_poles(A_red, B_red, desired_poles)
    K_pp = place_obj.gain_matrix

    print("Pole placement gain matrix K_pp:")
    print(K_pp)
    return (K_pp,)


@app.cell(hide_code=True)
def _(A_red, B_red, K_pp, np):
    A_cl = A_red - B_red @ K_pp
    eigvals = np.linalg.eigvals(A_cl)
    print("Closed-loop eigenvalues:", eigvals)
    return (A_cl,)


@app.cell(hide_code=True)
def _(A_cl, K_pp, np, plt, settling_indices):
    def _():
        from scipy.signal import StateSpace, lsim

        t = np.linspace(0, 40, 1000)
        x0 = np.array([1.0, 0.0, 0.1, 0.0])  # Initial state

        u = np.zeros_like(t)
        sys_cl = StateSpace(A_cl, np.zeros((4, 1)), np.eye(4), np.zeros((4, 1)))

        t, y, _ = lsim(sys_cl, U=u, T=t, X0=x0)

        plt.plot(t, y[:, 0], label='Δx (Lateral position error)')
        plt.xlabel('Time (s)')
        plt.ylabel('Δx (m)')
        plt.title('Closed-loop response of lateral position error')
        plt.legend()
        plt.grid()
        plt.show()

        # Constraint checks
        delta_theta = y[:, 2]  # Δθ(t)
        delta_phi = - (y @ K_pp.T).flatten()  # Control input Δφ(t)

        theta_constraint_satisfied = np.all(np.abs(delta_theta) < np.pi/2)
        phi_constraint_satisfied = np.all(np.abs(delta_phi) < np.pi/2)

        print(f"Constraint |Δθ(t)| < π/2 satisfied? {theta_constraint_satisfied}")
        print(f"Constraint |Δφ(t)| < π/2 satisfied? {phi_constraint_satisfied}")

        settling_indicess = np.where(np.abs(y[:, 0]) <= 0.1)[0]
        if len(settling_indices) > 0:
            settling_time = t[settling_indicess[0]]
            print(f"Δx settles within ±0.1 at t = {settling_time:.2f} s")
        else:
            return print("Δx does not settle within ±0.1 in the simulation time.")


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Optimal Control

    Using optimal, find a gain matrix $K_{oc}$ that satisfies the same set of requirements that the one defined using pole placement.

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Conception du Contrôleur Optimal (LQR)

    L’objectif est de stabiliser le système linéarisé autour de l’état d’équilibre avec :

    - \(\Delta \theta(0) = \frac{45}{180} \pi\) rad
    - \(\Delta \dot{\theta}(0) = 0\)
    - \(\Delta x(0) = 0\), \(\Delta \dot{x}(0) = 0\)

    La condition est que \(\Delta \theta(t)\) tende vers zéro en environ 20 secondes, tout en respectant :

    - \(|\Delta \theta(t)| < \frac{\pi}{2}\)
    - \(|\Delta \varphi(t)| < \frac{\pi}{2}\)

    Le drift sur \(\Delta x(t)\) n’est pas pris en compte.
    """
    )
    return


@app.cell
def _(np, plt):
    def LQR():
        from scipy.linalg import solve_continuous_are

        A = np.array([[0, 1, 0, 0],
                      [0, 0, -9.81, 0],
                      [0, 0, 0, 1],
                      [0, 0, 14.7, 0]])

        B = np.array([[0],
                      [0],
                      [0],
                      [1]])

        Q = np.diag([0, 0, 100, 1])
        R = np.array([[1]])

        P = solve_continuous_are(A, B, Q, R)
        K_oc = np.linalg.inv(R) @ B.T @ P

        A_cl = A - B @ K_oc

        x0 = np.array([0, 0, 45/180*np.pi, 0])
        dt = 0.01
        T = 25
        time = np.arange(0, T, dt)
        x = np.zeros((4, len(time)))
        x[:, 0] = x0

        for i in range(1, len(time)):
            x[:, i] = x[:, i-1] + dt * (A_cl @ x[:, i-1])

        u = -K_oc @ x

        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.plot(time, x[2,:])
        plt.axhline(np.pi/2, color='r', linestyle='--')
        plt.axhline(-np.pi/2, color='r', linestyle='--')
        plt.xlabel('Temps (s)')
        plt.ylabel('Angle Δθ (rad)')
        plt.title('Évolution de Δθ(t)')
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(time, u.flatten())
        plt.axhline(np.pi/2, color='r', linestyle='--')
        plt.axhline(-np.pi/2, color='r', linestyle='--')
        plt.xlabel('Temps (s)')
        plt.ylabel('Commande Δφ (rad)')
        plt.title('Évolution de la commande Δφ(t)')
        plt.grid(True)

        plt.tight_layout()
        return plt.show()


    LQR()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Validation

    Test the two control strategies (pole placement and optimal control) on the "true" (nonlinear) model and check that they achieve their goal. Otherwise, go back to the drawing board and tweak the design parameters until they do!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Validation sur le modèle non-linéaire

    On teste ici les contrôleurs précédemment conçus (**pole placement** et **LQR**) sur le modèle complet non-linéaire du booster.

    ### Critères à respecter :
    - $\theta(t)$ doit converger vers 0 sans dépasser $\pm \pi/2$
    - La commande $\phi(t)$ doit rester physiquement réaliste ($|\phi(t)| < \pi/2$)
    - Pour le contrôleur LQR, on veut aussi que $x(t)$ revienne à zéro

    ### Que vérifie-t-on ?
    Si le contrôleur conçu sur le modèle **linéarisé** fonctionne aussi sur le **vrai modèle**, c’est qu’il est robuste. Sinon… retour au tuning 

    """
    )
    return


@app.cell
def _(M, g, np, plt, redstart_solve):
    def validate_on_nonlinear(K, label="Pole Placement"):
        t_span = [0.0, 20.0]
        y0 = [0.0, 0.0, 0.0, 0.0, np.pi/4, 0.0]  # état initial : inclinaison 45°

        def f_phi_control(t, y):
            state_red = np.array([y[0], y[1], y[4], y[5]])  # [x, dx, theta, dtheta]
            delta_phi = -K @ state_red
            delta_phi = np.clip(delta_phi.item(), -np.pi/2, np.pi/2)  # convertir en scalaire
            return np.array([M * g, delta_phi])  # poussée constante, angle variable

        sol = redstart_solve(t_span, y0, f_phi_control)
        t = np.linspace(t_span[0], t_span[1], 1000)
        states = sol(t)

        x = states[0]
        theta = states[4]
        phi = np.array([
            (-K @ np.array([states[0][i], states[1][i], states[4][i], states[5][i]])).item()
            for i in range(len(t))
        ])
        phi = np.clip(phi, -np.pi/2, np.pi/2)

        # Tracés
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        plt.plot(t, x, label="x(t)")
        plt.title("Position latérale x(t)")
        plt.grid()
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(t, theta, label="θ(t)")
        plt.title("Inclinaison θ(t)")
        plt.axhline(np.pi/2, ls="--", color="red")
        plt.axhline(-np.pi/2, ls="--", color="red")
        plt.grid()
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(t, phi, label="φ(t)")
        plt.title("Commande φ(t)")
        plt.axhline(np.pi/2, ls="--", color="red")
        plt.axhline(-np.pi/2, ls="--", color="red")
        plt.grid()
        plt.legend()

        plt.suptitle(f"Validation du contrôleur ({label})")
        plt.tight_layout()
        plt.show()

    return (validate_on_nonlinear,)


@app.cell
def _(np):
    from scipy.linalg import solve_continuous_are

    # Renommage pour éviter les conflits
    A_lqr = np.array([[0, 1, 0, 0],
                      [0, 0, -9.81, 0],
                      [0, 0, 0, 1],
                      [0, 0, 14.7, 0]])

    B_lqr = np.array([[0],
                      [0],
                      [0],
                      [1]])

    Q = np.diag([0, 0, 100, 1])
    R_lqr = np.array([[1]])

    # Résolution de Riccati et gain
    P = solve_continuous_are(A_lqr, B_lqr, Q, R_lqr)
    K_oc = np.linalg.inv(R_lqr) @ B_lqr.T @ P

    return (K_oc,)


@app.cell
def _(K_oc, K_pp, validate_on_nonlinear):
    validate_on_nonlinear(K_pp, label="Pole Placement")
    validate_on_nonlinear(K_oc, label="LQR")
    return


if __name__ == "__main__":
    app.run()
