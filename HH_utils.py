import numpy as np
import time
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import ratinabox
from ratinabox.Neurons import Neurons
from ratinabox.utils import *

import tomplotlib as tpl

ratinabox.stylize_plots()
ratinabox.autosave_plots = True
ratinabox.figure_directory = "./figures/"

tpl.figure_directory = "./figures/"
tpl.set_colorscheme("viridis", 5)


class PyramidalNeurons(Neurons):
    """The PyramidalNeuorn class defines a layer of Neurons() whos firing rates are derived from the firing rates in two DendriticCompartments. They are theta modulated, during early theta phase the apical DendriticCompartment (self.apical_compartment) drives the soma, during late theta phases the basal DendriticCompartment (self.basal_compartment) drives the soma.

    Must be initialised with an Agent and a 'params' dictionary.

    Check that the input layers are all named differently.
    List of functions:
        • get_state()
        • update()
        • update_dendritic_compartments()
        • update_weights()
        • plot_loss()
        • plot_rate_map()
    """

    default_params = {
        "n": 10,
        "name": None,
        # theta params
        "theta_func": "square",  #'sine', 'square' etc.
        "theta_freq": 5,
        "theta_frac": 0.5,  # -->0 all basal input, -->1 all apical input
        "theta_phase_offset": 0,
        "activation_params": {
            "activation": "linear",
        },
        "dendrite_activation_params": {
            "activation": "linear"
        },  # you may want to set these manually for each compartment
    }

    def __init__(self, Agent, params={}):
        """Initialises a layer of pyramidal neurons

        Args:
            Agent (_type_): _description_
            params (dict, optional): _description_. Defaults to {}.
        """

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)
        super().__init__(Agent, self.params)

        self.history["loss"] = []

        self.basal_compartment = DendriticCompartment(
            self.Agent,
            params={
                "soma": self,
                "name": f"{self.name}_basal",
                "n": self.n,
                "color": self.color,
                "activation_params": self.dendrite_activation_params,
            },
        )
        self.apical_compartment = DendriticCompartment(
            self.Agent,
            params={
                "soma": self,
                "name": f"{self.name}_apical",
                "n": self.n,
                "color": self.color,
                "activation_params": self.dendrite_activation_params,
            },
        )

    def update(self):
        """Updates the firing rate of the layer. Saves a loss (lpf difference between basal and apical). Also adds noise."""
        self.update_dendritic_compartments()
        super().update()  # this sets and saves self.firingrate

        # update a smoothed history of the loss
        tau_smooth = 60
        fr_b, fr_a = (
            self.basal_compartment.firingrate,
            self.apical_compartment.firingrate,
        )
        loss = np.mean(np.abs(fr_b - fr_a))
        # loss = np.mean(((fr_b - fr_a)**2))
        dt = self.Agent.dt
        if self.Agent.t < 2 / self.theta_freq:
            self.loss = None
        else:
            self.loss = (dt / tau_smooth) * loss + (1 - dt / tau_smooth) * (
                self.loss or loss
            )
        self.history["loss"].append(self.loss)
        return

    def update_dendritic_compartments(self):
        """Individually updates the basal and apical firing rates."""
        self.basal_compartment.update()
        self.apical_compartment.update()
        return

    def get_state(self, evaluate_at="last", **kwargs):
        """Returns the firing rate of the soma. This depends on the firing rates of the basal and apical compartments and the current theta phase. By default the theta  is obtained from self.Agent.t but it can be passed manually as an kwarg to override this.

        theta (or theta_gating) is a number between [0,1] controlling flow of information into soma from the two compartment.s 0 = entirely basal. 1 = entirely apical. Between equals weighted combination. The function theta_gating() takes a time and returns theta.
        Args:
            evaluate_at (str, optional): 'last','agent','all' or None (in which case pos can be passed directly as a kwarg). Defaults to "last".
        Returns:
            firingrate
        """
        # theta can be passed in manually as a kwarg. If it isn't, the time from the agent will be used to get theta. Theta determines how much basal and how much apical this neurons uses. If a list if passed the first item from the list is used and the list and removed from the list
        if "theta" in kwargs:
            theta = kwargs["theta"]
            if type(theta) is list:
                theta_ = theta[0]
                del theta[0]
                if len(theta) == 1:
                    kwargs["theta"] = theta[0]
                else:
                    kwargs["theta"] = theta
            else:
                theta_ = theta
        else:
            theta_ = theta_gating(
                t=self.Agent.t,
                func=self.theta_func,
                freq=self.theta_freq,
                frac=self.theta_frac,
                phase_offset=self.theta_phase_offset,
            )
        fr_basal, fr_apical = 0, 0
        # these are special cases, no need to even get their fr's if they aren't used
        if (
            evaluate_at == "last"
        ):  # these have recently been updated, don't update them twice
            if theta_ != 0:
                fr_apical = self.apical_compartment.firingrate
            if theta_ != 1:
                fr_basal = self.basal_compartment.firingrate
        else:
            if theta_ != 0:
                fr_apical = self.apical_compartment.get_state(evaluate_at, **kwargs)
            if theta_ != 1:
                fr_basal = self.basal_compartment.get_state(evaluate_at, **kwargs)
        firingrate = (1 - theta_) * fr_basal + (theta_) * fr_apical
        firingrate = activate(firingrate, other_args=self.activation_params)
        return firingrate

    def update_weights(self):
        """Trains the weights, this function actually defined in the dendrite class."""
        # if self.Agent.t > 2/self.theta_freq:
        self.basal_compartment.update_weights()
        self.apical_compartment.update_weights()
        return

    def plot_loss(self, fig=None, ax=None):
        """Plots the loss against time to see if learning working"""
        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=(1.5, 1.5))
            ax.set_yscale("log")
            ax.spines["right"].set_color("none")
            ax.spines["top"].set_color("none")
        t = np.array(self.history["t"]) / 60
        loss = np.array(self.history["loss"])
        t, loss = t[loss != None], loss[loss != None]
        loss = np.array(loss, dtype=float)
        ax.plot(t, loss, color=self.color, label=self.name)
        ax.set_xlim(left=0)
        ax.legend(frameon=False)
        ax.set_xlabel("Training time / min")
        ax.set_ylabel("Loss")
        return fig, ax

    def plot_rate_map(self, route="basal", **kwargs):
        """This is a wrapper function for the general Neuron class function plot_rate_map.
        For pyramidal neurons it is not enough just to "plot" a rate map, you must also specify the theta phase, i.e. which compartments to draw from. It takes the same arguments as Neurons.plot_rate_map() but, in addition, route can be set to basal or apical in which case theta is set correspondingly and the soma with take its input from downstream or upstream sources entirely.

        The arguments for the standard plottiong function plot_rate_map() can be passed as usual as kwargs.

        Args:
            route (str, optional): _description_. Defaults to 'basal'.
        """
        if route is None:
            fig, ax = super().plot_rate_map(**kwargs, method="history")
        else:
            if route == "basal":
                theta = 0
            if route == "apical":
                theta = [1, 0]
            fig, ax = super().plot_rate_map(**kwargs, theta=theta)
        return fig, ax


class DendriticCompartment(Neurons):
    """The DendriticCompartment class defines a layer of Neurons() whos firing rates are an activated linear combination of input layers. This class is a subclass of Neurons() and inherits it properties/plotting functions.

    Must be initialised with an Agent and a 'params' dictionary.
    Input params dictionary must  contain a list of input_layers which feed into these Neurons. This list looks like [Neurons1, Neurons2,...] where each is a Neurons() class.

    Currently supported activations include 'sigmoid' (paramterised by max_fr, min_fr, mid_x, width), 'relu' (gain, threshold) and 'linear' specified with the "activation_params" dictionary in the inout params dictionary. See also activate() for full details.

    Check that the input layers are all named differently.
    List of functions:
        • get_state()
        • add_input()
    """

    default_params = {
        "soma": None,
        "tau": None,
        "noise_scale": 0.01,
        "noise_coherence_time": 0.3,
        "add_noise": True,
        "target_layer": None,
        "activation_params": {"activation": "linear"},
    }

    def __init__(self, Agent, params={}):
        self.Agent = Agent
        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)
        super().__init__(Agent, self.params)

        self.V = np.zeros(self.n)
        self.inputs = {}
        self.noise = np.zeros(self.n)
        self.global_excitation = 0.0

        if self.target_layer is None:
            self.target_layer = self.soma

    def add_input(
        self,
        input_layer,
        eta=0.01,
        w_init=1,
        w_bias=0,
        bias=False,
        L1=0.0,
        L2=0.0,
        tau_PI=100e-3,
        recurrent_loop=False,
        dales_law=False,
    ):
        """Adds an input layer to the class. Each input layer is stored in a dictionary of self.inputs. Each has an associated matrix of weights which are initialised randomly.

        Args:
            input_layer (_type_): the layer which feeds into this compartment
            eta: learning rate of the weights
            w_init: initialisation scale of the weights
            L1: how much L1 regularisation
            L2: how much L2 regularisation
            tau_PI: smoothing timescale of plasticity induction variable
            recurrent_loop: whether to ignore this input on rate map evaluation
        """
        name = input_layer.name
        n_in = input_layer.n
        # w has size n+1 if bias is true
        w = np.random.normal(
            loc=0, scale=w_init / np.sqrt(n_in), size=(self.n, n_in + bias)
        )
        w += w_bias / np.sqrt(n_in)
        if bias == True:
            w[:, -1] = np.zeros(
                self.n
            )  # biases initialise at zero to prevent explosions

        if dales_law is not False:
            if dales_law == "inh":
                w = w * (w < 0)
            if dales_law == "exc":
                w = w * (w > 0)
        I = np.zeros(n_in + bias)
        PI = np.zeros_like(w)
        if name in self.inputs.keys():
            print(f"There already exists a layer called {name}. Overwriting it now.")
        self.inputs[name] = {}
        self.inputs[name]["layer"] = input_layer
        self.inputs[name]["w"] = w
        self.inputs[name]["bias"] = bias
        self.inputs[name]["I"] = I  # input current
        self.inputs[name]["I_temp"] = None  # input current
        self.inputs[name]["PI"] = PI  # plasticity induction variable
        self.inputs[name]["eta"] = eta
        self.inputs[name]["L2"] = L2
        self.inputs[name]["L1"] = L1
        self.inputs[name]["dales_law"] = dales_law

        self.inputs[name]["tau_PI"] = tau_PI
        self.inputs[name]["recurrent_loop"] = recurrent_loop

    def get_state(self, evaluate_at="last", return_V=False, **kwargs):
        """Returns the "firing rate" of the dendritic compartment. By default this layer uses the last saved firingrate from its input layers. Alternatively evaluate_at and kwargs can be set to be anything else which will just be passed to the input layer for evaluation.
        Once the firing rate of the inout layers is established these are multiplied by the weight matrices and then activated to obtain the firing rate of this FeedForwardLayer.

        Args:
            evaluate_at (str, optional). Defaults to 'last'.
        Returns:
            firingrate: array of firing rates
        """
        if evaluate_at == "last":
            V = np.zeros(self.n)
        elif evaluate_at == "all":
            V = np.zeros(
                (self.n, self.Agent.Environment.flattened_discrete_coords.shape[0])
            )
        else:
            V = np.zeros((self.n, kwargs["pos"].shape[0]))
        for inputlayer in self.inputs.values():
            w = inputlayer["w"]
            if evaluate_at == "last":
                I = inputlayer["layer"].firingrate
            else:  # kick can down the road let input layer decide how to evaluate the firingrate
                if (
                    inputlayer["recurrent_loop"] == False
                ):  # ignore recurrent loops to avoid recursion error
                    I = inputlayer["layer"].get_state(evaluate_at, **kwargs)
                else:
                    I = None

            if I is not None:
                if (
                    inputlayer["bias"] == True
                ):  # append single 1 (or row of ones) to current
                    I = np.concatenate(
                        (I, np.ones_like(I[0]).reshape((1,) + I[0].shape))
                    )
                inputlayer["I_temp"] = I
                V += np.matmul(w, I)
        V += self.global_excitation
        if return_V == True:
            return V
        else:
            firingrate = activate(V, other_args=self.activation_params)
            return firingrate

    def update(self):
        """Updates firingrate of this compartment and saves it to file"""
        # update firing rate
        V = self.get_state(return_V=True)
        if self.tau is not None:
            self.V = (
                (1 - self.Agent.dt / self.tau) * self.V + (self.Agent.dt / self.tau) * V
            ).reshape(-1)
        else:
            self.V = V.reshape(-1)

        # convert to firing rate
        self.firingrate = activate(self.V, other_args=self.activation_params)
        self.firingrate_deriv = activate(
            self.V, other_args=self.activation_params, deriv=True
        )

        # add noise
        dt = self.Agent.dt
        self.noise += ornstein_uhlenbeck(
            dt,
            self.noise,
            drift=0.0,
            noise_scale=self.noise_scale,
            coherence_time=self.noise_coherence_time,
        )

        if self.add_noise == True:
            self.firingrate += self.noise

        # update the input currents needed for learning
        for inputlayer in self.inputs.values():
            inputlayer["I"] = inputlayer["I_temp"].reshape(-1)
        self.save_to_history()
        return

    def update_weights(self):
        """Implements the weight update: dendritic prediction of somatic activity."""
        target = self.target_layer.firingrate

        delta = (target - self.firingrate) * (self.firingrate_deriv)
        dt = self.Agent.dt
        for inputlayer in self.inputs.values():
            eta = inputlayer["eta"]
            if eta != 0:
                tau_PI = inputlayer["tau_PI"]
                # assert (dt / tau_PI) < 0.2
                I = inputlayer["I"]
                w = inputlayer["w"].copy()
                L2 = inputlayer["L2"]
                L1 = inputlayer["L1"]
                if True and tau_PI != 0:
                    # first updates plasticity induction variable (smoothed delta error outer product with the input current for this input layer)
                    PI_old = inputlayer["PI"]
                    PI_update = np.outer(delta, I)
                    PI = (dt / tau_PI) * PI_update + (1 - dt / tau_PI) * PI_old
                    inputlayer["PI"] = PI
                else:
                    PI = np.outer(delta, I)

                dw = eta * (PI - L2 * w - L1 * np.sign(w))
                w = w + dw
                if inputlayer["dales_law"] == "inh":
                    w = w * (w <= 0)
                elif inputlayer["dales_law"] == "exc":
                    w = w * (w >= 0)
                inputlayer["w"] = w
        return

    def plot_weights(self, colormatch=False):
        """Plots the incoming weights to this layer
        Args:
            colormatch (bool, optional): etermines whether to match colouring scales for different weight matrices. Defaults to False.
        Returns:
            fig, ax
        """
        data = []
        for name, info in self.inputs.items():
            w = info["w"]
            data.append((w, name))
        fig = plt.figure(figsize=(9, 9))
        ax = ImageGrid(
            fig,
            222,
            nrows_ncols=(1, len(data)),
            axes_pad=0.1,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="5%",
            cbar_pad=0.1,
        )
        cax = ax.cbar_axes[0]
        ax = np.array(ax)
        # fig, ax = plt.subplots(1,len(data),figsize=(len(data)*3,3),sharey=True,gridspec_kw={'width_ratios': [datum[0].shape[1]/datum[0].shape[0] for datum in data]})
        vmin, vmax = round(np.min(data[0][0]), 3), round(np.max(data[0][0]), 3)
        if colormatch == True:
            vmin, vmax = np.min(data[0][0]), np.max(data[0][0])
            for i, datum in enumerate(data):
                vmin, vmax = min(vmin, np.min(datum[0])), max(vmax, np.max(datum[0]))
            vmax, vmin = max(vmax, -vmin), min(vmin, -vmax)
        for i, datum in enumerate(data):
            d = datum[0]
            if colormatch == True:
                scale = min(vmin / np.min(d), vmax / np.max(d))
                d *= scale
                ax[i].text(x=0.9, y=0.9, s=f"x {scale:.1f}", fontdict={"size": 3})
            im = ax[i].imshow(d, aspect=1, vmin=vmin, vmax=vmax, cmap="viridis")
            ax[i].set_xlabel(datum[1])
            ax[i].set_xticks([0, d.shape[1]])
            for spine in ax[i].spines.values():
                spine.set_visible(False)
            if i > 0:
                ax[i].set_yticks([])

        ax[0].set_yticks([0, self.n])
        ax[0].set_ylabel(self.name)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_ticks([0, vmin, vmax])
        cbar.set_ticklabels([0, vmin, vmax])
        cbar.outline.set_visible(False)
        return fig, ax


def theta_gating(t, func="square", freq=10, frac=0.5, phase_offset=0):
    """Returns the theta gating \in [0,1] depending on the time. Set the frequency and frac spend in 1 or 0 state

    Args:
        t (_type_): _description_
        func: type of theta oscillation ('square', 'sine' etc.)
        freq (int, optional): _description_. Defaults to 10.
        frac (float, optional): _description_. Defaults to 0.5.
        phase: phase offset from 0 to 1

    Returns:
        _type_: _description_
    """
    T = 1 / freq
    if func == "square":
        phase = (((t / T) % 1) + phase_offset) % 1
        if phase < frac:
            return 1
        elif phase >= frac:
            return 0

    if func == "sine":
        n = np.log(0.5) / np.log(frac)
        phase = (((t / T) % 1) ** n + phase_offset) % 1
        return 0.5 * (1 + np.sin(2 * np.pi * phase))

    if func[:8] == "constant":
        return float(func[9:])


def get_loss(layer=[]):
    """Returns to loss summed over many layers.

    Args:
        layers (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """
    if layer.__class__.__name__ == "PyramidalNeurons":
        loss = layer.loss or 0
        return loss
    else:
        return None


def run_simulation(
    time_min=10,
    Agent=None,
    layers=[],
    train=True,
    no_plot=False,
    force_theta=None,
    verbose=True,
):
    """Takes a list of layers and determines whether runs a big simulation"""
    from tqdm import tqdm

    last_update = Agent.t

    T = np.zeros(1 + len(layers))

    if force_theta is not None:
        old_thetas = []
        for layer in layers:
            if "theta_func" in layer.__dict__.keys():
                old_thetas.append(layer.theta_func)
                layer.theta_func = force_theta
    try:
        if verbose:
            pbar = tqdm(range(int(60 * time_min / Agent.dt)))
        else:
            pbar = range(int(60 * time_min / Agent.dt))
        for _ in pbar:
            t0 = time.time()
            Agent.update()
            t1 = time.time()
            T[0] += t1 - t0
            for i, layer in enumerate(layers):
                t0 = time.time()
                layer.update()
                if layer.__class__.__name__ == "PyramidalNeurons":
                    if train == True:
                        layer.update_weights()
                t1 = time.time()
                T[i + 1] += t1 - t0
            if Agent.t > last_update + 10:
                desc = []
                for layer in layers:
                    L = get_loss(layer)
                    if L is not None:
                        desc.append((layer.name, f"{L:.2e}"))
                if verbose:
                    pbar.set_description(f"<Loss> = {desc}")
                last_update = Agent.t
    except KeyboardInterrupt:
        pass

    if train:
        Agent.t_finished_training = Agent.t

    if force_theta is not None:
        for layer in layers:
            if "theta_func" in layer.__dict__.keys():
                layer.theta_func = old_thetas.pop(0)

    fig, ax = None, None
    if no_plot != True:
        for layer in layers:
            if layer.__class__.__name__ == "PyramidalNeurons":
                fig, ax = layer.plot_loss(fig=fig, ax=ax)
    return fig, ax


def train_decoder(layer, t_start=None, t_end=None, method="gaussian_process", fps=10):
    from sklearn.linear_model import Ridge
    from sklearn.gaussian_process import GaussianProcessRegressor

    skip = max(1, int((1 / fps) / layer.Agent.dt))
    t = np.array(layer.history["t"])
    if t_start is None:
        i_start = 0
    else:
        i_start = np.argmin(np.abs(t - t_start))
    if t_end is None:
        i_end = -1
    else:
        i_end = np.argmin(np.abs(t - t_end))
    t = t[i_start:i_end][::skip]
    fr = np.array(layer.history["firingrate"])[i_start:i_end][::skip]
    pos = np.array(layer.Agent.history["pos"])[i_start:i_end][::skip]
    if (layer.Agent.Environment.dimensionality == "1D") and (
        layer.Agent.Environment.boundary_conditions == "periodic"
    ):
        # first transform x into a more "periodic" variable
        r = layer.Agent.Environment.scale / (2 * np.pi)
        cos_theta = np.cos(pos[:, 0] / r)
        sin_theta = np.sin(pos[:, 0] / r)
        pos = np.stack((cos_theta, sin_theta)).T
    if method == "linear_regression":
        model = Ridge(alpha=0.01)
    if method == "argmax":
        model = Argmax()
    if method == "gaussian_process":
        model = GaussianProcessRegressor(random_state=10)
    model.fit(fr, pos)
    layer.decoding_model = model
    return


def decode_position(
    layer, decoder=None, t_start=None, t_end=None, return_true_pos=False
):
    t = np.array(layer.history["t"])
    if t_start is None:
        i_start = 0
    else:
        i_start = np.argmin(np.abs(t - t_start))
    if t_end is None:
        i_end = -1
    else:
        i_end = np.argmin(np.abs(t - t_end))
    t = t[i_start:i_end]
    fr = np.array(layer.history["firingrate"])[i_start:i_end]
    pos = np.array(layer.Agent.history["pos"])[i_start:i_end]

    if decoder is None:
        decoder = layer.decoding_model
    if (layer.Agent.Environment.dimensionality == "1D") and (
        layer.Agent.Environment.boundary_conditions == "periodic"
    ):
        r = layer.Agent.Environment.scale / (2 * np.pi)
        decoded_position = decoder.predict(fr)
        decoded_position = r * np.arctan2(
            decoded_position[:, 1], decoded_position[:, 0]
        )
        decoded_position[decoded_position < 0] = (
            2 * np.pi * r + decoded_position[decoded_position < 0]
        )
    else:
        decoded_position = decoder.predict(fr)
    if return_true_pos:
        return t, decoded_position, pos.reshape(-1)
    else:
        return t, decoded_position


def run_pathint_test(
    layers, decoding_layer, spin_up_secs=10, test_secs=20, n_tests=1, plot=True
):
    """Runs a path integration test: First the Agent undergo a "spin-up" where the network is run in normal (oscillatory) mode for a period. Then it is tested, the network is placed in generative mode and tries to path integrate for a period of time. The decoding layer specifies which layer a decoder will be trained on."""

    # train a decoder on the last 10 mins of training data
    Ag = layers[0].Agent
    train_decoder(
        decoding_layer,
        t_start=Ag.t_finished_training - 60 * 10,
        t_end=Ag.t_finished_training,
    )
    errors = []
    for i in tqdm(range(n_tests)):
        # Generate test data
        # Ag.speed_std = 0.15
        # Ag.speed_coherence_time = 1
        for layer in layers:
            layer.theta_func = "square"
            layer.add_noise = False
        # 10 seconds of spin up
        run_simulation(
            time_min=spin_up_secs / 60,
            Agent=Ag,
            layers=layers,
            train=False,
            no_plot=True,
            verbose=False,
        )
        for layer in layers:
            layer.theta_func = "constant_1"
            layer.add_noise = False
        run_simulation(
            time_min=test_secs / 60,
            Agent=Ag,
            layers=layers,
            train=False,
            no_plot=True,
            verbose=False,
        )
        t, decoded_pos, true_pos = decode_position(
            decoding_layer,
            decoder=decoding_layer.decoding_model,
            t_start=Ag.t - test_secs - spin_up_secs,
            return_true_pos=True,
        )
        # ensure same size (can be an off by 1 error):
        true_pos = true_pos[: t.shape[0]]
        decoded_pos = decoded_pos[: t.shape[0]]
        error = np.diag(
            Ag.Environment.get_distances_between___accounting_for_environment(
                decoded_pos.reshape(-1, 1), true_pos.reshape(-1, 1)
            )
        )

        errors.append(list(error))
        t = t - t[0]

    errors = np.array(errors).reshape(n_tests, t.shape[0])
    t_start = Ag.t - test_secs - spin_up_secs

    if plot:
        # plot rate time series with decoding ontop of it
        fig, ax = decoding_layer.plot_rate_timeseries(
            t_start=t_start, imshow=True, autosave=False
        )
        ax.axvline(x=Ag.t - test_secs, c="r", linestyle=":")
        # slice = Ag.get_history_slice(t_start=Ag.t-test_mins*60,framerate=10)
        # time, pos = Ag.history['t'][slice], Ag.history['pos'][slice]
        print(t.shape, true_pos.shape, decoded_pos.shape)
        ax.scatter(t + t_start, true_pos, c="w")
        ax.scatter(t + t_start, decoded_pos, c="g")
        ax.set_title("Activity time series of the decoding neurons")
        tpl.save_figure(fig, "pathint")

        # plot just the decoding results
        fig, ax = plt.subplots(
            figsize=(
                ratinabox.MOUNTAIN_PLOT_WIDTH_MM / 25,
                0.5 * ratinabox.MOUNTAIN_PLOT_WIDTH_MM / 25,
            )
        )
        ax.scatter(
            (t + t_start) / 60,
            decoded_pos,
            c="C1",
            linewidth=0,
            alpha=0.5,
            s=5,
            label="Decoded",
        )
        ax.scatter(
            (t + t_start) / 60,
            true_pos,
            c="C0",
            linewidth=0,
            alpha=0.5,
            s=5,
            label="True",
        )
        ax.set_title("True vs decoded performance")
        ax.axvline(x=(Ag.t - test_secs) / 60, c="r", linestyle=":")
        plt.legend()
        tpl.save_figure(fig, "decoding")

        # plot decoding errors
        fig, ax = plt.subplots(
            figsize=(
                ratinabox.MOUNTAIN_PLOT_WIDTH_MM / 25,
                0.5 * ratinabox.MOUNTAIN_PLOT_WIDTH_MM / 25,
            )
        )
        mean = np.mean(errors, axis=0)
        sem = np.std(errors, axis=0) / np.sqrt(n_tests)
        ax.plot(t, mean, c="r")
        if n_tests > 1:
            ax.fill_between(t, mean + sem, mean - sem, facecolor="r", alpha=0.5)
        ax.set_ylim(top=Ag.Environment.extent[-1] / 2)
        ax.axvline(spin_up_secs, c="r", linestyle=":")
        ax.set_title("Error in decoding position")
        tpl.save_figure(fig, "error")

    return t, errors


def cross_cor(x1, x2):
    """x1 = aray like (Nx1, Nt)
       x2 = aray like (Nx2, Nt)
    returns cross correlation matrix Nx1 x Nx2
    """
    Nt = x1.shape[-1]
    assert x2.shape[-1] == Nt
    Nx1 = x1.shape[0]
    Nx2 = x2.shape[0]

    x1 = x1 - x1.mean(axis=1).reshape(-1, 1)
    x2 = x2 - x2.mean(axis=1).reshape(-1, 1)

    x1 = x1.reshape((Nx1, 1, Nt))
    x2 = x2.reshape((1, Nx2, Nt))

    Cij = (x1 * x2).sum(axis=2)
    Cii = (x1 * x1).sum(axis=2)
    Cjj = (x2 * x2).sum(axis=2)
    CiiCjj = np.sqrt(Cii * Cjj)

    return Cij / CiiCjj


class GaussianProcessSampler:
    def __init__(self, tau=2, dt=0.1, C=None, Cinv=None):
        self.tau = tau
        self.dt = dt

        if C is None:
            self.t_range = np.arange(0, 3 * tau, dt)
            self.C_Nplus1 = RBF_kernel(
                self.t_range, self.t_range, self.tau
            ) + 0.00001 * np.identity(len(self.t_range + 1))
        else:
            self.C_Nplus1 = C
        self.C_N = self.C_Nplus1[:-1, :-1]

        if Cinv is None:
            self.C_Ninv = np.linalg.inv(self.C_N)
        else:
            self.C_Ninv = Cinv

        self.N = len(self.C_Nplus1) - 1

        self.k = self.C_Nplus1[:-1, -1]
        self.kappa = self.C_Nplus1[-1, -1]

        self.var = self.kappa - self.k.T @ (self.C_Ninv @ self.k) #Mackay eqn 45.43
        self.k_onCinv = self.k.T @ self.C_Ninv 

        self.T = np.array([0])
        self.X = np.array([0])

    def sample_next(self):
        """Takes the last N samples (padded with zeros if not existent yet) and samples next point from gaussian posterior. We are follow the method outlined in Mackay Information theory chapter 45"""
        past_targets = self.X[-self.N :]
        N_missing = self.N - len(past_targets)
        past_targets = np.pad(past_targets, (N_missing, 0))

        mean = self.k_onCinv @ past_targets #Mackay eqn 45.42

        next_point = np.random.normal(mean, np.sqrt(self.var))
        self.X = np.append(self.X, next_point)
        self.T = np.append(self.T, self.T[-1] + self.dt)
        return next_point


def RBF_kernel(x1, x2, tau=1):
    """returns gram matrix of shape (len(x1),len(x2))"""
    x1 = np.array(x1).reshape(-1, 1)
    x2 = np.array(x1).reshape(1, -1)
    return np.exp(-((x1 - x2) ** 2) / (2 * tau**2))


class GaussianProcessNeurons(Neurons):
    """Each neuron has a firing rate sampled from a Gaussian process.
    You set the timescale of each one.

    Args:
        Neurons (_type_): _description_
    """

    default_params = {
        "timescales": 1.0,
        "offset": 0,
    }

    def __init__(self, Agent, params={}):
        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        super().__init__(Agent, params=self.params)

        if type(self.timescales) in [float, int]:
            self.timescales = np.array([self.timescales] * self.n)
        else:
            self.n = len(self.timescales)

        self.samplers = [
            GaussianProcessSampler(self.timescales[i], dt=self.Agent.dt)
            for i in range(self.n)
        ]

    def get_state(self, evaluate_at="now"):
        if evaluate_at == "now":
            firingrate = np.array([sampler.sample_next() for sampler in self.samplers])
            firingrate += self.offset
            return firingrate
        else:
            return None


def periodic_plot(ax, t, x, color="C1", **kwargs):
    abs_diff = np.abs(np.diff(x))
    mask = np.hstack([abs_diff > abs_diff.mean() + 3 * abs_diff.std(), [False]])
    masked_x = np.ma.MaskedArray(x, mask)
    ax.plot(t, masked_x, c=color, alpha=0.5, linewidth=2)

    return


def align_matrix(M):
    M_align = M.copy()
    for i, row in enumerate(M):
        M_align[i] = np.roll(M[i], shift=-i)
    M_align = np.roll(M_align, shift=int(M.shape[1] / 2), axis=1)
    return M_align


def autocorrelations(true, hidden, max_lag=2):
    # plots autocorrelation of true and hidden firing rates for the last 50 seconds of data collected 
    Ag = true.Agent
    t = np.array(Ag.history["t"])
    dt = Ag.dt
    id_start = np.argmin(np.abs(t - (Ag.t - 50)))

    true_data = np.array(true.history["firingrate"])[id_start:, :].T
    generated_data = np.array(hidden.history["firingrate"])[id_start:, :].T

    fig, ax = plt.subplots()
    ax.set_xlabel("Time lag / s")
    ax.set_ylabel("Autocorrelation")
    tpl.xy_axes(ax)
    ax.set_xlim(right=max_lag)
    ax.set_xticks([0, max_lag])
    for data, color in zip([true_data, generated_data], [true.color, hidden.color]):
        tau_now = 0
        tau = []
        cors = []
        for i in range(int(max_lag / dt)):
            tau.append(tau_now)
            if i == 0:
                cor = np.diagonal(cross_cor(data[:, :], data[:, :]))
            else:
                cor = np.diagonal(cross_cor(data[:, i:], data[:, :-i]))
            tau_now += dt
            cors.append(list(cor))
        tau = np.array(tau)
        cors = np.array(cors)
        for j in range(data.shape[0]):
            ax.plot(tau, cors[:, j], c=color, alpha=0.1)
        ax.plot(tau, np.mean(cors, axis=1), linewidth=2, c=color)
    return fig, ax
