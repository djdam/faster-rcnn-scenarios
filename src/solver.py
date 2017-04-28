import caffe.proto.caffe_pb2 as caffe_pb2
from os.path import join
from caffe_config import CaffeConfig
from to_string import to_string

class SolverConfig(CaffeConfig):

    def __init__(self, base_lr=0.001, lr_policy="step", solver_type="SGD", step_size=10000, display=20, momentum=0.9, gamma=0.1, weight_decay=5e-4):
        CaffeConfig.__init__(self)
        self.base_lr=base_lr
        self.solver_type=solver_type
        self.step_size=step_size
        self.display=display
        self.momentum=momentum
        self.gamma=gamma
        self.weight_decay=weight_decay
        self.lr_policy=lr_policy

    def path(self):
        self.dir_exists_or_create()

        return join(self.scen_dir, 'stage%s_%s_solver.prototxt' % (self.stage, self.net_type))

    def generate(self, net):
        if net == None:
            raise Exception("Net not provided!")

        self.setScenario(net.scenarios_dir, net.scenario)
        self.stage=net.stage
        self.net_type=net.network_type

        s = caffe_pb2.SolverParameter()

        # Specify locations of the train and (maybe) test networks.
        s.train_net = net.path()

        s.lr_policy=self.lr_policy

        # The number of iterations over which to average the gradient.
        # Effectively boosts the training batch size by the given factor, without
        # affecting memory utilization.
        s.iter_size = 1

        # Solve using the stochastic gradient descent (SGD) algorithm.
        # Other choices include 'Adam' and 'RMSProp'.
        s.type = self.solver_type

        # Set the initial learning rate for SGD.
        s.base_lr = self.base_lr

        # Set `lr_policy` to define how the learning rate changes during training.
        # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
        # every `stepsize` iterations.
        s.gamma = self.gamma
        s.stepsize = self.step_size

        # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
        # weighted average of the current gradient and previous gradients to make
        # learning more stable. L2 weight decay regularizes learning, to help prevent
        # the model from overfitting.
        s.momentum = self.momentum
        s.weight_decay = self.weight_decay

        # Display the current training loss and accuracy every 1000 iterations.
        s.display = 20

        # Snapshots are files used to store networks we've trained.  Here, we'll
        # snapshot every 10K iterations -- ten times during training.
        s.snapshot = 0

        # Train on the GPU.  Using the CPU to train large networks is very slow.
        s.solver_mode = caffe_pb2.SolverParameter.GPU

        return self.save(s)

    def __repr__(self):
        return to_string(self)

def generate(solver_config):
    solver_config.stage=1
    solver_config.stage = 1

    solver_config.generate()



run=0