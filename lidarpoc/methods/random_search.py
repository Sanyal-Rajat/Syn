from scipy.stats import qmc


def optimize(objective, seed=0, **_):
    for candidate in qmc.LatinHypercube(objective.dim, seed=seed).random(objective.remaining):
        objective(candidate)
