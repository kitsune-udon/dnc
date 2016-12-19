# Differentiable Neural Computer
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

def var_fzero(shape):
    return chainer.Variable(np.zeros(shape, dtype=np.float32))

def var_fones(shape):
    return chainer.Variable(np.ones(shape, dtype=np.float32))

class RecurrentBlock(chainer.Chain):
    def __init__(self, n_units_of_input, n_units_of_hidden_layer):
        super(RecurrentBlock, self).__init__()
        self.n_units_of_input = n_units_of_input
        self.n_units_of_hidden_layer = n_units_of_hidden_layer
        n_in, n_out = n_units_of_input + 2 * n_units_of_hidden_layer, n_units_of_hidden_layer
        self.add_link('linear_i', L.Linear(n_in, n_out))
        self.add_link('linear_f', L.Linear(n_in, n_out))
        self.add_link('linear_s', L.Linear(n_in, n_out))
        self.add_link('linear_o', L.Linear(n_in, n_out))
        self.reset_state()

    def __call__(self, input_vector, h_in):
        batch_size = input_vector.shape[0]
        if self.min_batch_size is None:
            self.min_batch_size = batch_size
        else:
            assert(batch_size <= self.min_batch_size)
            self.max_batch_size = batch_size

        if self.h is None:
            h_prev = var_fzero((batch_size, self.n_units_of_hidden_layer))
        else:
            h_prev = self.h[0:batch_size]

        if h_in is None:
            h_in = var_fzero((batch_size, self.n_units_of_hidden_layer))
        else:
            assert(batch_size == h_in.shape[0])

        v = F.hstack([input_vector, h_prev, h_in])
        i = F.sigmoid(self.linear_i(v))
        f = F.sigmoid(self.linear_f(v))
        s = f * self.state + i * F.tanh(self.linear_s(v))
        o = F.sigmoid(self.linear_o(v))
        h = o * F.tanh(s)

        self.state = s
        self.h = h

        return h

    def reset_state(self):
        self.h = None
        self.state = 0
        self.min_batch_size = None

class RecurrentBlockDummy(chainer.Chain): # NOTE : This Link has no memory ability.
    def __init__(self, n_units_of_input, n_units_of_hidden_layer):
        super(RecurrentBlockDummy, self).__init__()
        n_in, n_out = n_units_of_input + n_units_of_hidden_layer, n_units_of_hidden_layer
        self.n_units_of_hidden_layer = n_units_of_hidden_layer
        self.add_link('l1', L.Linear(n_in, n_out))

    def __call__(self, input_vector, h_in):
        batch_size = input_vector.shape[0]

        if h_in is None:
            h_in = var_fzero((batch_size, self.n_units_of_hidden_layer))
        else:
            assert(h_in.shape[0] >= batch_size)
            h_in = h_in[0:batch_size]

        v = F.hstack([input_vector, h_in])
        h = F.tanh(self.l1(v))

        return h

class ControllerSubfunctions:
    @staticmethod
    def oneplus(x):
        return 1 + F.log(1 + F.exp(x))

    @staticmethod
    def extract_params(xi, width, n_read_heads):
        batch_size = xi.shape[0]
        offs = 0
        read_keys = F.reshape(xi[:, offs:offs+width*n_read_heads], (batch_size, n_read_heads, width))
        offs += width * n_read_heads
        read_strengths = ControllerSubfunctions.oneplus(F.reshape(xi[:, offs:offs+n_read_heads], (batch_size, n_read_heads)))
        offs += n_read_heads
        write_key = xi[:, offs:offs+width]
        offs += width
        write_strength = ControllerSubfunctions.oneplus(xi[:, offs])
        offs += 1
        erase_vector = F.sigmoid(xi[:, offs:offs+width])
        offs += width
        write_vector = xi[:, offs:offs+width]
        offs += width
        free_gates = F.sigmoid(F.reshape(xi[:, offs:offs+n_read_heads], (batch_size, n_read_heads)))
        offs += n_read_heads
        allocation_gate = F.sigmoid(xi[:, offs])
        write_gate = F.sigmoid(xi[:, offs+1])
        offs += 2
        read_modes = F.reshape(F.softmax(F.reshape(xi[:, offs:offs+3*n_read_heads], (-1, 3))), (batch_size, n_read_heads, 3))
        return (read_keys, read_strengths, write_key, write_strength,
                erase_vector, write_vector, free_gates, allocation_gate,
                write_gate, read_modes)

    @staticmethod
    def read_vectors(memory, read_weightings):
        """ (batch_size,n_locations,width) -> (batch_size,n_read_heads,n_locations) -> (batch_size,n_read_heads,width) """
        return F.batch_matmul(read_weightings, memory)

    @staticmethod
    def updated_memory(memory, write_weighting, erase_vector, write_vector):
        """ (batch_size,n_locations,width) -> (batch_size,n_locations) -> (batch_size,width) -> (batch_size,width) -> (batch_size,n_locations,width) """
        mem, w, e, v = memory, write_weighting, erase_vector, write_vector
        r = mem * (1 - F.batch_matmul(w, e, transb=True)) + F.batch_matmul(w, v, transb=True)
        return r

    @staticmethod
    def content_based_addressing(memory, keys, strengths):
        """ (M,n_locations,width) -> (M,N,width) -> (M,N) -> (M,N,n_locations) """
        M, n_locations, width = memory.shape
        N = keys.shape[1]
        m, k, s = memory, keys, strengths
        m = F.reshape(F.normalize(F.reshape(m, (-1, width))), (M, n_locations, width))
        k = F.reshape(F.normalize(F.reshape(k, (-1, width))), (M, N, width))
        t = F.scale(F.batch_matmul(k, m, transb=True), s, axis=0)
        r = F.reshape(F.softmax(F.reshape(t, (-1, n_locations))), (M, N, n_locations))
        return r


    @staticmethod
    def memory_retention_vector(free_gates, read_weightings_prev):
        """ (batch_size,n_read_heads) -> (batch_size,n_read_heads,n_locations) -> (batch_size,n_locations) """
        batch_size, n_read_heads = free_gates.shape
        n_locations = read_weightings_prev.shape[2]
        t = 1 - F.scale(read_weightings_prev, free_gates, axis=0)
        r = var_fones((batch_size, n_locations))
        for i in range(n_read_heads):
            r *= t[:, i, :]
        return r

    @staticmethod
    def updated_usage_vector(usage_vector_prev, write_weighting_prev, psi):
        """ (batch_size,n_locations) -> (batch_size,n_locations) -> (batch_size,n_locations) -> (batch_size,n_locations) """
        u, w = usage_vector_prev, write_weighting_prev
        r = (u + w - u * w) * psi
        return r

    @staticmethod
    def allocation_weighting(usage_vector): # NOTE: not differentiable
        """ (batch_size,n_locations) -> (batch_size,n_locations) """
        batch_size, n_locations = usage_vector.shape
        u = usage_vector.data
        phi = np.argsort(u, axis=1)
        s = np.ones(batch_size, dtype=np.float32)
        a = np.zeros((batch_size, n_locations), dtype=np.float32)
        asc = np.arange(batch_size).astype(np.int32)
        for j in range(n_locations):
            k = phi[:, j]
            t = u[asc, k]
            a[asc, k] = (1 - t) * s
            s *= t
        return a

    @staticmethod
    def write_weighting(allocation_gate, write_gate, allocation_weighting, write_content_weighting):
        """ (batch_size,) -> (batch_size,) -> (batch_size,n_locations) -> (batch_size,n_locations) -> (batch_size,n_locations) """
        ga, gw, a, c = allocation_gate, write_gate, allocation_weighting, write_content_weighting
        r =  F.scale((F.scale(a, ga, axis=0) + F.scale(c, 1 - ga, axis=0)), gw, axis=0)
        return r

    @staticmethod
    def precedence_weighting(write_weighting, precedence_weighting_prev):
        """ (batch_size,n_locations) -> (batch_size,n_locations) -> (batch_size,n_locations) """
        w, p = write_weighting, precedence_weighting_prev
        r = F.scale(p, (1 - F.sum(w, axis=1)), axis=0) + w
        return r

    @staticmethod
    def updated_link_matrix(link_matrix_prev, write_weighting, precedence_weighting_prev):
        """ (batch_size,n_locations,n_locations) -> (batch_size,n_locations) -> (batch_size,n_locations) -> (batch_size,n_locations,n_locations) """
        l0, w, p = link_matrix_prev, write_weighting, precedence_weighting_prev
        batch_size, n_locations = w.shape[0], w.shape[1]
        wrep = F.broadcast_to(w[:, :, None], l0.shape)
        l = (1 - wrep - F.transpose(wrep, (0, 2, 1))) * l0 + F.batch_matmul(w, p, transb=True)
        r = l * (1 - np.eye(n_locations, dtype=np.float32))
        return r

    @staticmethod
    def forward_weightings(link_matrix, read_weightings_prev):
        """ (batch_size,n_locations,n_locations) -> (batch_size,n_read_heads,n_locations) -> (batch_size,n_read_heads,n_locations) """
        return F.batch_matmul(read_weightings_prev, link_matrix, transb=True)

    @staticmethod
    def backward_weightings(link_matrix, read_weightings_prev):
        """ (batch_size,n_locations,n_locations) -> (batch_size,n_read_heads,n_locations) -> (batch_size,n_read_heads,n_locations) """
        return F.batch_matmul(read_weightings_prev, link_matrix)

    @staticmethod
    def write_content_weighting(memory_prev, write_key, write_strength):
        """ (batch_size,n_locations,width) -> (batch_size,width) -> (batch_size,) -> (batch_size,n_locations) """
        batch_size, n_locations = memory_prev.shape[0], memory_prev.shape[1]
        c = ControllerSubfunctions.content_based_addressing(memory_prev, write_key[:, None, :], write_strength[:, None])
        r = F.reshape(c, (batch_size, n_locations))
        return r

    @staticmethod
    def read_content_weightings(memory, read_keys, read_strengths):
        """ (batch_size,n_locations,width) -> (batch_size,n_read_heads,width) -> (batch_size,n_read_heads) -> (batch_size,n_read_heads,n_locations) """
        return ControllerSubfunctions.content_based_addressing(memory, read_keys, read_strengths)

    @staticmethod
    def read_weightings(read_modes, backward_weightings, read_content_weightings, forward_weightings):
        """ (batch_size,n_read_heads,3) -> (batch_size,n_read_heads,n_locations) -> (batch_size,n_read_heads,n_locations) -> (batch_size,n_read_heads,n_locations) -> (batch_size,n_read_heads,n_locations) """
        pi, b, c, f = read_modes, backward_weightings, read_content_weightings, forward_weightings
        x = F.scale(b, pi[:, :, 0], axis=0)
        y = F.scale(c, pi[:, :, 1], axis=0)
        z = F.scale(f, pi[:, :, 2], axis=0)
        return x + y + z

    @staticmethod
    def controller_input_vector(x, read_vectors):
        batch_size = x.shape[0]
        return F.hstack([x, F.reshape(read_vectors, (batch_size, -1))])

    @staticmethod
    def join_hs(hs):
        return F.hstack(hs)

    @staticmethod
    def join_read_vectors(read_vectors):
        batch_size = read_vectors.shape[0]
        return F.reshape(read_vectors, (batch_size, -1))

class DefaultController(chainer.Chain):
    def __init__(self, controller_input_vector_dim, output_dim, interface_vector_dim, n_units_of_hidden_layer, n_layers, norecurrent=False):
        super(DefaultController, self).__init__()
        self.add_link('hidden_layers', chainer.ChainList())
        self.add_link('linear_yps', L.Linear(n_layers * n_units_of_hidden_layer, output_dim, nobias=True))
        self.add_link('linear_xi', L.Linear(n_layers * n_units_of_hidden_layer, interface_vector_dim, nobias=True))

        if norecurrent:
            hidden_layer_class = RecurrentBlockDummy
        else:
            hidden_layer_class = RecurrentBlock

        for i in range(n_layers):
            self.hidden_layers.add_link(hidden_layer_class(controller_input_vector_dim, n_units_of_hidden_layer))

        self.n_layers = n_layers
        self.reset_state()

    def __call__(self, chi):
        f = ControllerSubfunctions

        hs = [None] * (self.n_layers+1)
        for i in range(self.n_layers):
            hs[i+1] = self.hidden_layers[i](chi, hs[i+1])

        hs_joined = f.join_hs(hs[1:])
        ypsilon = self.linear_yps(hs_joined)
        xi = self.linear_xi(hs_joined)
        return (ypsilon, xi)

    def reset_state(self):
        if hasattr(self.hidden_layers[0], 'reset_state'):
            for i in range(self.n_layers):
                self.hidden_layers[i].reset_state()


class Core(chainer.Chain):
    def __init__(self, n_locations, memory_width, n_read_heads, output_dim, controller):
        super(Core, self).__init__()
        self.add_link('controller', controller)
        self.add_link('linear_r', L.Linear(memory_width * n_read_heads, output_dim, nobias=True))
        self.n_locations = n_locations
        self.memory_width = memory_width
        self.n_read_heads = n_read_heads
        self.reset_state()

    def __call__(self, x):
        f = ControllerSubfunctions
        batch_size = x.shape[0]
        if self.min_batch_size is None:
            self.min_batch_size = batch_size
        else:
            assert(batch_size <= self.min_batch_size)
            self.min_batch_size = batch_size

        if self.read_vectors is None:
            read_vectors = var_fzero((batch_size, self.n_read_heads, self.memory_width))
        else:
            read_vectors = self.read_vectors[0:batch_size]

        chi = f.controller_input_vector(x, read_vectors)
        ypsilon, xi = self.controller(chi)
        rvs_joined = f.join_read_vectors(read_vectors)
        y = ypsilon + self.linear_r(rvs_joined)
        self.process_interface_vector(xi)

        return y

    def process_interface_vector(self, xi):
        f = ControllerSubfunctions
        batch_size = xi.shape[0]
        (read_keys, read_strengths, write_key, write_strength,
         erase_vector, write_vector, free_gates, allocation_gate,
         write_gate, read_modes) = f.extract_params(xi, self.memory_width, self.n_read_heads)

        if self.ws_read is None:
            ws_read_prev = var_fzero((batch_size, self.n_read_heads, self.n_locations))
        else:
            ws_read_prev = self.ws_read[0:batch_size]

        if self.w_write is None:
            w_write_prev = var_fzero((batch_size, self.n_locations))
        else:
            w_write_prev = self.w_write[0:batch_size]

        if self.p is None:
            p_prev = var_fzero((batch_size, self.n_locations))
        else:
            p_prev = self.p[0:batch_size]

        if self.memory is None:
            memory_prev = var_fzero((batch_size, self.n_locations, self.memory_width))
        else:
            memory_prev = self.memory[0:batch_size]

        if self.link_matrix is None:
            link_matrix_prev = var_fzero((batch_size, self.n_locations, self.n_locations))
        else:
            link_matrix_prev = self.link_matrix[0:batch_size]

        if self.u is None:
            u_prev = var_fzero((batch_size, self.n_locations))
        else:
            u_prev = self.u[0:batch_size]

        psi = f.memory_retention_vector(free_gates, ws_read_prev)
        u = f.updated_usage_vector(u_prev, w_write_prev, psi)
        a = f.allocation_weighting(u)
        c_w = f.write_content_weighting(memory_prev, write_key, write_strength)
        w_write= f.write_weighting(allocation_gate, write_gate, a, c_w)
        p = f.precedence_weighting(w_write, p_prev)
        link_matrix = f.updated_link_matrix(link_matrix_prev, w_write, p_prev)
        fs = f.forward_weightings(link_matrix, ws_read_prev)
        bs = f.backward_weightings(link_matrix, ws_read_prev)
        memory = f.updated_memory(memory_prev, w_write, erase_vector, write_vector)
        cs_r = f.read_content_weightings(memory, read_keys, read_strengths)
        ws_read = f.read_weightings(read_modes, bs, cs_r, fs)
        vs_read = f.read_vectors(memory, ws_read)

        self.ws_read = ws_read
        self.w_write = w_write
        self.memory = memory
        self.p = p
        self.link_matrix = link_matrix
        self.read_vectors = vs_read
        self.u = u

    def reset_state(self):
        self.min_batch_size = None
        self.read_vectors = None
        self.ws_read = None
        self.w_write = None
        self.p = None
        self.memory = None
        self.link_matrix = None
        self.u = None
        if hasattr(self.controller, 'reset_state'):
            self.controller.reset_state()

class DefaultDNC(chainer.Chain):
    def __init__(self, n_locations, memory_width, n_read_heads, input_dim, output_dim, n_units_of_hidden_layer, n_layers, norecurrent=False):
        super(DefaultDNC, self).__init__()
        controller_input_vector_dim = input_dim + memory_width * n_read_heads
        interface_vector_dim = (memory_width * n_read_heads) + 3 * memory_width + 5 * n_read_heads + 3
        controller = DefaultController(controller_input_vector_dim, output_dim, interface_vector_dim, n_units_of_hidden_layer, n_layers, norecurrent)
        self.add_link('core', Core(n_locations, memory_width, n_read_heads, output_dim, controller))

    def __call__(self, x):
        return self.core(x)

    def reset_state(self):
        self.core.reset_state()
