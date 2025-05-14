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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell
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
    return FFMpegWriter, FuncAnimation, np, plt, sci, tqdm


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


@app.cell
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return


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


@app.cell
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
    (mo.video(src=_filename))
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


@app.cell
def _():
    g = 1      # m/s²
    M = 1      # kg
    l = 1         # m
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


@app.function
def reactor_force(f, theta, phi):
    import numpy as np  # import local, pas de conflit avec les autres cellules
    beta = theta + phi
    fx = -f * np.sin(beta)
    fy = f * np.cos(beta)
    return np.array([fx, fy])


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Center of Mass

    Give the ordinary differential equation that governs $(x, y)$.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Équation du centre de masse du booster

    Le booster est soumis à deux forces :

    - La gravité, qui agit vers le bas :

    $$
    \vec{F}_g = \begin{bmatrix} 0 \\\\ -Mg \end{bmatrix}
    $$

    - La poussée du moteur, appliquée à la base du booster.  
      Elle est orientée d’un angle $\varphi$ par rapport à l’axe du booster, lequel est incliné d’un angle $\theta$ par rapport à la verticale.  
      L’angle total de la poussée dans le repère global est donc $\theta + \varphi$ :

    $$
    \vec{F}_r = f \begin{bmatrix} -\sin(\theta + \varphi) \\\\ \cos(\theta + \varphi) \end{bmatrix}
    $$

    D’après la deuxième loi de Newton appliquée au centre de masse $\vec{r}(t) = (x(t), y(t))$ :

    $$
    M \ddot{\vec{r}} = \vec{F}_g + \vec{F}_r
    $$

    On obtient le système d’équations différentielles :

    $$
    \begin{cases}
    \ddot{x}(t) = -\dfrac{f}{M} \cdot \sin(\theta + \varphi) \\\\
    \ddot{y}(t) = \dfrac{f}{M} \cdot \cos(\theta + \varphi) - g
    \end{cases}
    $$

    #### 🔢 Cas du modèle simplifié

    Avec les constantes numériques du modèle :
    - $M = 1$ (kg)  
    - $g = 1$ (m/s²)

    Le système devient :


    $$
    \begin{aligned}
    \ddot{x}(t) &= -f \cdot \sin(\theta + \varphi) \\
    \ddot{y}(t) &= f \cdot \cos(\theta + \varphi) - 1
    \end{aligned}
    $$
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
def _(mo):
    mo.md(
        r"""

    \[
    J = \frac{1}{12} M L_{\text{total}}^2
    \]

    In our case, the total length of the booster is \( L_{\text{total}} = 2\ell \). Substituting this into the formula:

    \[
    J = \frac{1}{3} M \ell^2
    \]
    """
    )
    return


@app.cell
def _(M, l):
    J = (1/3) * M * (l**2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Tilt

    Give the ordinary differential equation that governs the tilt angle $\theta$.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Le mouvement de rotation du booster autour de son centre de masse est gouverné par le *principe fondamental de la dynamique en rotation* :

    $$
    J \cdot \ddot{\theta}(t) = \tau(t)
    $$

    où :

    - $J$ est le *moment d’inertie* du booster par rapport à son centre,
    - $\ddot{\theta}(t)$ est l’*accélération angulaire*,
    - $\tau(t)$ est le *moment (ou couple)* appliqué par la poussée du réacteur.

    ---
    #### 1. Vecteur de position $\vec{r}$

    Le point d’application de la poussée est la *base du booster*, située à une distance $\ell$ sous le centre de masse.

    Le vecteur $\vec{r}$ qui va *du centre de masse vers la base* est :

    $$
    \vec{r} = -\ell \cdot \vec{u}_\theta
    = -\ell \begin{bmatrix} \sin(\theta) \\ \cos(\theta) \end{bmatrix}
    $$

    ---

    #### 2. Vecteur force $\vec{F}$

    La poussée est orientée à un angle $\varphi$ par rapport à l’axe du booster, et donc à un angle total $\theta + \varphi$ dans le repère global.

    Sa projection est :

    $$
    \vec{F} = f \cdot \begin{bmatrix}
    - \sin(\theta + \varphi) \\
    \cos(\theta + \varphi)
    \end{bmatrix}
    $$
    ---

    #### 3. Calcul du produit vectoriel en 2D

    Le moment est donné par :

    $$
    \tau = r_x F_y - r_y F_x
    $$

    avec :
    - $r_x = -\ell \sin(\theta)$
    - $r_y = -\ell \cos(\theta)$
    - $F_x = -f \sin(\theta + \varphi)$
    - $F_y = f \cos(\theta + \varphi)$

    \[
    \begin{aligned}
    \tau &= (-\ell \sin(\theta)) \cdot f \cos(\theta + \varphi)
         - (-\ell \cos(\theta)) \cdot (-f \sin(\theta + \varphi)) \\
         &= -\ell f \sin(\theta) \cos(\theta + \varphi)
         - \ell f \cos(\theta) \sin(\theta + \varphi)
    \end{aligned}
    \]

    On utilise l’identité trigonométrique :

    $$
    \sin(a + b) = \sin a \cos b + \cos a \sin b
    $$

    donc :

    $$
    \tau = -\ell f \cdot \sin(\theta + \varphi)
    $$

    ---
    En utilisant le principe fondamental :

    $$
    J \cdot \ddot{\theta}(t) = -\ell f \cdot \sin(\theta + \varphi)
    $$

    et en remplaçant $J = \dfrac{1}{3} M \ell^2$ :

    $$
    \frac{1}{3} M \ell^2 \cdot \ddot{\theta}(t) = -\ell f \cdot \sin(\theta + \varphi)
    $$

    On simplifie :

    $$
    \ddot{\theta}(t) = -\frac{3f}{M\ell} \cdot \sin(\theta + \varphi)
    $$

    ---

    ### Cas du modèle

    Avec :
    $M = 1$
     $\ell = 1$

    On obtient :

    $$
    \boxed{\ddot{\theta}(t) = -3f \cdot \sin(\theta + \varphi)}
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


@app.cell
def _(M, g, l, np, plt, sci):
    def simulate_booster_drop(f_val=0.0, phi_val=0.0):
        def f_phi(t, y):
            return np.array([f_val, phi_val])

        def redstart_ode(t, y):
            x, dx, y_pos, dy, theta, dtheta = y
            f, phi = f_phi(t, y)
            ddx = f * np.sin(theta + phi) / M
            ddy = (f * np.cos(theta + phi) - M * g) / M
            ddtheta = -3 * f * np.sin(theta + phi) / (M * l)
            return [dx, ddx, dy, ddy, dtheta, ddtheta]

        sol = sci.solve_ivp(redstart_ode, [0.0, 5.0], [0.0, 0.0, 10.0, 0.0, 0.0, 0.0], t_eval=np.linspace(0, 5, 1000))
        t = sol.t
        y_pos = sol.y[2]
        plt.plot(t, y_pos)
        plt.title("Chute  du booster")
        plt.xlabel("temps (s)")
        plt.ylabel("hauteur $y(t)$")
        plt.grid(True)
        plt.show()
    return (simulate_booster_drop,)


@app.cell
def _(simulate_booster_drop):
    simulate_booster_drop()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0)$, can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell
def _(M, g, np, plt, sci):
    # Constants
    y0 = 10.0     # initial position (m)
    vy0 = -2.0     # initial velocity (m/s)
    yf = 1.0      # final position (m)
    vyf = 0.0     # final velocity (m/s)
    T = 5.0       # total time (s)

    # Cubic polynomial: y(t) = a0 + a1*t + a2*t^2 + a3*t^3
    # Boundary conditions:
    # y(0) = y0, y'(0) = vy0, y(T) = yf, y'(T) = vyf
    A = np.array([
        [1, 0,    0,      0],
        [0, 1,    0,      0],
        [1, T,  T*2,  T*3],
        [0, 1,  2*T,  3*T**2]
    ])
    b = np.array([y0, vy0, yf, vyf])
    a0, a1, a2, a3 = np.linalg.solve(A, b)

    # Acceleration: y''(t) = 2*a2 + 6*a3*t
    def acceleration(t):
        return 2*a2 + 6*a3*t

    def force(t):
        return M * acceleration(t) + M * g

    # System dynamics: [y, vy]
    def dynamics(t, y):
        return [y[1], acceleration(t)]

    # Time vector
    t_eval = np.linspace(0, T, 500)

    # Solve ODE
    sol = sci.solve_ivp(dynamics, [0, T], [y0, vy0], t_eval=t_eval)

    # Compute force over time
    f_t = force(t_eval)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axs[0].plot(sol.t, sol.y[0], label='y(t)')
    axs[0].set_ylabel('Position (m)')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(sol.t, sol.y[1], label="y'(t)", color='orange')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(t_eval, f_t, label='f(t)', color='green')
    axs[2].set_ylabel('Force (N)')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend()
    axs[2].grid()

    plt.tight_layout()
    plt.show()
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
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualization

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


if __name__ == "__main__":
    app.run()
