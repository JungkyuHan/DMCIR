import numpy as np
import math


class DDPMUtil:
    def __init__(self, T, clip_min=-1.0, clip_max=1.0):
        self.T = T
        self.pi_alphas = None    # a^{-}_{t} = Pi_{s=1}^{t} a_{s}
        self.one_minus_pi_alphas = None   # 1 - a^{-}_{t}
        self.sqrt_pi_alphas = None    # sqrt(a^{-}_{t})
        self.sqrt_one_minus_pi_alphas = None  # sqrt(1-a^{-}_{t})
        self.alphas = None          # a_{t}
        self.sqrt_alphas = None     # sqrt(a_{t})
        self.betas = None           # b_{t} = (1-a_{t})
        self.p_vars = None            # var(t) = b_{t} * (1 - a^{-}_{t-1}) / ((1 - a^{-}_{t-1}))
        self.p_sigmas = None          # sigma(t) = sqrt(var(t))
        self.loss_coef_x0 = None      # 1/(2*var(t)) * (a^{-}_{t-1} * b_{t}^2) / (1 - a^{-}_{t})^2
        self.loss_coef_eps = None
        self.clip_min = clip_min
        self.clip_max = clip_max

    @staticmethod
    def linear_beta_func(ts, end_ts, start_beta=0.0001, end_beta=0.02, noise_scale=1.0):
        start = noise_scale * start_beta
        end = noise_scale * end_beta
        step = (end - start) / end_ts
        delta = (ts - 1) * step
        beta = start + delta
        return beta

    @staticmethod
    def cosine_beta_func(ts, end_ts):
        if ts == 0:
            return 1.0
        i_4_zero = -1.0/(end_ts - 1.0)
        i_4_t = (ts - 1.0) / (end_ts - 1.0)
        i_4_t_minus_1 = (ts - 1.0 - 1.0) / (end_ts - 1.0)
        f_0 = math.cos((i_4_zero + 0.008) / 1.008 * math.pi / 2) ** 2
        f_i = math.cos((i_4_t + 0.008) / 1.008 * math.pi / 2) ** 2
        f_i_minus_1 = math.cos((i_4_t_minus_1 + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar_t = f_i / f_0
        alpha_bar_t_minus_1 = f_i_minus_1 / f_0
        beta_t = min(1.0-alpha_bar_t/alpha_bar_t_minus_1, 0.999)
        return beta_t

    @staticmethod
    def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].

        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                          produces the cumulative product of (1-beta) up to that
                          part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                         prevent singularities.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    @staticmethod
    def betas_from_linear_variance(steps, variance, max_beta=0.999):
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
        return np.array(betas)

    @staticmethod
    def beta_func2(ts, end_ts, start_beta=0.0001, end_beta=0.02, noise_scale=1.0, mode="linear"):
        """
        Given the schedule name, create the betas for the diffusion process.
        """
        if mode == "linear" or mode == "linear-var":
            start = noise_scale * start_beta
            end = noise_scale * end_beta
            if mode == "linear":
                return np.linspace(start, end, ts, dtype=np.float32)
            else:
                return DDPMUtil.betas_from_linear_variance(end_ts, np.linspace(start, end, ts, dtype=np.float32))
        elif mode == "cosine":
            return DDPMUtil.betas_for_alpha_bar(end_ts,
                                                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
        elif mode == "binomial":  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
            ts = np.arange(end_ts)
            betas = [1 / (end_ts - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {mode}!")

    def gen_noise_schedule(self, start_beta=0.0001, end_beta=0.02, noise_scale=1.0, mode="linear"):
        self.alphas = [1.0]          # a_{t}
        self.sqrt_alphas = [1.0]     # sqrt(a_{t})
        self.betas = [0.0]          # b_{t} = (1-a_{t})
        self.pi_alphas = [1.0]    # a^{-}_{t} = Pi_{s=1}^{t} a_{s}
        self.one_minus_pi_alphas = [0.0001]   # 1 - a^{-}_{t}
        self.sqrt_pi_alphas = [1.0]    # sqrt(a^{-}_{t})
        self.sqrt_one_minus_pi_alphas = [0.0]  # sqrt (1-a^{-}_{t})
        self.p_vars = [0.0]           # var(t) = b_{t} * (1 - a^{-}_{t-1}) / ((1 - a^{-}_{t-1}))
        self.p_sigmas = [0.0]          # sigma(t) = sqrt(var(t))
        self.loss_coef_x0 = [1.0]  # 1/(2*var(t)) * (a^{-}_{t-1} * b_{t}^2) / (1 - a^{-}_{t})^2
        self.loss_coef_eps = [1.0]
        pi_alpha_t = self.pi_alphas[0]
        if mode == "cosine":
            beta_ts = DDPMUtil.beta_func2(0, self.T, start_beta=start_beta, end_beta=end_beta, noise_scale=noise_scale,
                                     mode="cosine")
        for ts in range(1, self.T+1):
            if mode == "cosine":
                # beta_t = beta_ts[ts-1]
                beta_t = DDPMUtil.cosine_beta_func(ts, self.T)
            else:
                beta_t = DDPMUtil.linear_beta_func(ts, self.T, start_beta=start_beta, end_beta=end_beta, noise_scale=noise_scale)
            alpha_t = 1.0 - beta_t
            pi_alpha_t *= alpha_t
            one_minus_pi_alpha_t = 1.0 - pi_alpha_t
            sqrt_alpha_t = np.sqrt(alpha_t)
            sqrt_pi_alpha_t = np.sqrt(pi_alpha_t)
            sqrt_one_minus_pi_alpha_t = np.sqrt(one_minus_pi_alpha_t)

            self.alphas.append(alpha_t)
            self.sqrt_alphas.append(sqrt_alpha_t)
            self.betas.append(beta_t)
            self.pi_alphas.append(pi_alpha_t)
            self.one_minus_pi_alphas.append(one_minus_pi_alpha_t)
            self.sqrt_pi_alphas.append(sqrt_pi_alpha_t)
            self.sqrt_one_minus_pi_alphas.append(sqrt_one_minus_pi_alpha_t)

            var_t = (self.betas[ts] * self.one_minus_pi_alphas[ts-1]) / self.one_minus_pi_alphas[ts]
            sd_t = np.sqrt(var_t)
            self.p_vars.append(var_t)
            self.p_sigmas.append(sd_t)
            '''
            if ts == 1:
                loss_coef_x0_t = 1.0
                loss_coef_eps_t = 1.0
            else:
                loss_coef_x0_t = (1.0 / (2 * var_t)) * (self.pi_alphas[ts-1] * (beta_t * beta_t)
                                                        / (one_minus_pi_alpha_t * one_minus_pi_alpha_t))
                loss_coef_eps_t = 0.5 * (self.pi_alphas[ts-1]/self.one_minus_pi_alphas[ts-1]
                                         - self.pi_alphas[ts]/self.one_minus_pi_alphas[ts])
            '''
            loss_coef_x0_t = (1.0 / (2 * var_t)) * (self.pi_alphas[ts-1] * (beta_t * beta_t)
                                                    / (one_minus_pi_alpha_t * one_minus_pi_alpha_t))
            loss_coef_eps_t = 0.5 * (self.pi_alphas[ts-1]/self.one_minus_pi_alphas[ts-1]
                                     - self.pi_alphas[ts]/self.one_minus_pi_alphas[ts])

            self.loss_coef_x0.append(loss_coef_x0_t)
            self.loss_coef_eps.append(loss_coef_eps_t)

        self.alphas = np.array(self.alphas, dtype=np.float32)
        self.sqrt_alphas = np.array(self.sqrt_alphas, dtype=np.float32)
        self.betas = np.array(self.betas, dtype=np.float32)
        self.pi_alphas = np.array(self.pi_alphas, dtype=np.float32)
        self.one_minus_pi_alphas = np.array(self.one_minus_pi_alphas, dtype=np.float32)
        self.sqrt_pi_alphas = np.array(self.sqrt_pi_alphas, dtype=np.float32)
        self.sqrt_one_minus_pi_alphas = np.array(self.sqrt_one_minus_pi_alphas, dtype=np.float32)
        self.p_vars = np.array(self.p_vars, dtype=np.float32)
        self.p_sigmas = np.array(self.p_sigmas, dtype=np.float32)
        self.loss_coef_x0 = np.array(self.loss_coef_x0, dtype=np.float32)
        self.loss_coef_eps = np.array(self.loss_coef_eps, dtype=np.float32)

        print("Noise schedule generation complete:")
        print("self.alphas.shape")
        print(self.alphas.shape)
        print("self.sqrt_alphas.shape")
        print(self.sqrt_alphas.shape)
        print("self.betas.shape")
        print(self.betas.shape)
        print("self.pi_alphas.shape")
        print(self.pi_alphas.shape)
        print("self.one_minus_pi_alphas.shape")
        print(self.one_minus_pi_alphas.shape)
        print("self.sqrt_pi_alphas.shape")
        print(self.sqrt_pi_alphas.shape)
        print("self.sqrt_one_minus_pi_alphas.shape")
        print(self.sqrt_one_minus_pi_alphas.shape)
        print("self.p_vars.shape")
        print(self.p_vars.shape)
        print("self.p_sigmas.shape")
        print(self.p_sigmas.shape)
        print("self.loss_coef_x0.shape")
        print(self.loss_coef_x0.shape)


    def param_shape(self, x_shape):
        x_shape_dim = len(x_shape)
        param_shape = [-1]
        for i in range(1, x_shape_dim):
            param_shape.append(1)
        param_shape = tuple(param_shape)
        return param_shape

    ###
    ## x0.shape = (batch_size, datadim1, datadim2, datadim3,...)
    ## t = (batch_size, t_for data) : int
    ## eps      = (batch_size, datadim1, datadim2, datadim3,...)
    def q_sample(self, x_0, t, eps, param_shape):
        sqrt_pi_alpha_t = self.sqrt_pi_alphas[t]
        sqrt_one_minus_pi_alpha_t = self.sqrt_one_minus_pi_alphas[t]
        sqrt_pi_alpha_t = np.reshape(sqrt_pi_alpha_t, param_shape)
        sqrt_one_minus_pi_alpha_t = np.reshape(sqrt_one_minus_pi_alpha_t, param_shape)

        x_t_means = np.multiply(sqrt_pi_alpha_t, x_0)
        x_t_deviations = np.multiply(sqrt_one_minus_pi_alpha_t, eps)
        x_t = x_t_means + x_t_deviations

        return x_t

    ###
    ## xt.shape = (batch_size, datadim1, datadim2, datadim3,...)
    ## t = (batch_size, t_for data) : int
    ## eps      = (batch_size, datadim1, datadim2, datadim3,...)
    def p_sample_eps_lb(self, x_t, t, eps_pred, noise, param_shape, clip=True):
        sqrt_alpha_t = self.sqrt_alphas[t]
        coef1 = 1.0/sqrt_alpha_t
        coef1 = coef1.astype(np.float32)
        beta_t = self.betas[t]
        sqrt_one_minus_pi_alpha_t = self.sqrt_one_minus_pi_alphas[t]
        coef2 = beta_t/sqrt_one_minus_pi_alpha_t
        sigma_t = self.p_sigmas[t]

        coef2 = np.reshape(coef2, param_shape)
        coef1 = np.reshape(coef1, param_shape)
        sigma_t = np.reshape(sigma_t, param_shape)
        p_mean = coef1 * (x_t - coef2 * eps_pred)

        x_prev = p_mean + sigma_t * noise

        if clip is True:
            x_prev = np.clip(x_prev, self.clip_min, self.clip_max)

        return x_prev

    ###
    ## xt.shape = (batch_size, datadim1, datadim2, datadim3,...)
    ## t = (batch_size, t_for data) : int
    ## eps      = (batch_size, datadim1, datadim2, datadim3,...)
    def p_sample_x0_lb(self, x_t, t, x0_pred, noise, param_shape, clip=True):
        sqrt_alpha_t = self.sqrt_alphas[t]
        sqrt_alpha_t_minus_1 = self.sqrt_alphas[t-1]
        one_minus_pi_alpha_t_minus_1 = self.one_minus_pi_alphas[t-1]
        one_minus_pi_alpha_t = self.one_minus_pi_alphas[t]
        inv_one_minus_pi_alpha_t = 1.0/one_minus_pi_alpha_t
        inv_one_minus_pi_alpha_t = inv_one_minus_pi_alpha_t.astype(np.float32)
        beta_t = self.betas[t]
        sigma_t = self.p_sigmas[t]

        inv_one_minus_pi_alpha_t = np.reshape(inv_one_minus_pi_alpha_t, param_shape)
        coef1 = sqrt_alpha_t * one_minus_pi_alpha_t_minus_1
        coef1 = np.reshape(coef1, param_shape)
        coef2 = sqrt_alpha_t_minus_1 * beta_t
        coef2 = np.reshape(coef2, param_shape)
        sigma_t = np.reshape(sigma_t, param_shape)
        p_mean = inv_one_minus_pi_alpha_t * (coef1 * x_t + coef2 * x0_pred)

        x_prev = p_mean + sigma_t * noise

        if clip is True:
            x_prev = np.clip(x_prev, self.clip_min, self.clip_max)

        return x_prev

