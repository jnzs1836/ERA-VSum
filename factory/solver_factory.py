from solvers import GANSolver, WGANSolver, SupervisedSolver, GroundTruthSolver, TestSolver
from exceptions import InvalidSolverException, InvalidModelException
from .model_factory import build_summarizer, build_compressor, build_discriminator, build_critic
import pickle


def open_pickle_file(filename):
    fp = open(filename, "rb")
    data = pickle.load(fp)
    return data
    with (open(filename, "rb")) as openfile:
        t = pickle.load(openfile)
        while True:
            try:
                t = pickle.load(openfile)
                print(t)
                return t
            except EOFError:
                print(EOFError)
                break


def get_difference_attention(dataset):
    return open_pickle_file(dataset)
    # with open("summe_diff_attention.pickle", "rb") as fp:
    #    return pickle.load(fp)

def build_gan_solver(config):
    solver = GANSolver(config)
    summarizer = build_summarizer(config)
    discriminator = build_discriminator(config)
    compressor = build_compressor(config)
    solver.build(compressor, summarizer, discriminator)
    return solver


def build_wgan_solver(config):
    solver = WGANSolver(config)
    summarizer = build_summarizer(config)
    critic = build_critic(config)
    compressor = build_compressor(config)
    solver.build(compressor, summarizer, critic)
    return solver


def build_supervised_solver(config):
    solver = SupervisedSolver(config)
    summarizer = build_summarizer(config, supervised=True)
    if config.compressing_features:
        linear_compressor = build_compressor(config)
    else:
        linear_compressor = None
    solver.build(summarizer, linear_compressor)
    return solver


def build_test_solver(config):
    solver = TestSolver(config)
    summarizer = build_summarizer(config)
    if config.solver == "GAN":
        critic = build_discriminator(config)
    else:
        critic = build_critic(config)
    compressor = build_compressor(config)
    solver.build(compressor, summarizer, critic)
    return solver


def build_solver(config):
    if config.solver == "GAN":
        return build_gan_solver(config)
    elif config.solver == "WGAN":
        return build_wgan_solver(config)
    elif config.solver == "Supervised":
        return build_supervised_solver(config)
    else:
        raise InvalidSolverException(config.solver)
