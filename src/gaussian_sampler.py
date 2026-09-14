import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from typing import Tuple, Dict, Any, Optional

# Empirical defaults from Jülich Pedestrian Dynamics Literature:
# Trait 1: Walking Speed (m/s) ~ Normal(1.34, std=0.20) -> Var = 0.04
# Trait 2: Reaction Time (s) ~ Normal(0.50, std=0.10)  -> Var = 0.01
# Covariance: 0.015 (positive correlation: faster movers tend to react faster or panic more intensely)
MU_DEFAULT = np.array([1.34, 0.50], dtype=np.float64)
SIGMA_DEFAULT = np.array([
    [0.04, 0.015],
    [0.015, 0.010]
], dtype=np.float64)


def sample_agents(n_agents: int = 1000, 
                  mu: np.ndarray = MU_DEFAULT, 
                  sigma: np.ndarray = SIGMA_DEFAULT, 
                  seed: Optional[int] = None) -> np.ndarray:
    """
    Samples N agent trait vectors x ~ N(mu, Sigma) using numpy.random.default_rng.
    
    Returns:
        agents: Array of shape (n_agents, 2) where column 0 is walking speed (m/s)
                and column 1 is reaction time (s).
    """
    rng = np.random.default_rng(seed)
    agents = rng.multivariate_normal(mu, sigma, size=n_agents)
    return agents


def validate_sample_statistics(agents: np.ndarray, 
                                mu: np.ndarray = MU_DEFAULT, 
                                sigma: np.ndarray = SIGMA_DEFAULT) -> Dict[str, Any]:
    """
    Validates that sampled population statistics converge to the theoretical mu and Sigma.
    """
    sample_mu = np.mean(agents, axis=0)
    sample_sigma = np.cov(agents, rowvar=False)

    abs_diff_mu = np.abs(sample_mu - mu)
    rel_diff_mu = abs_diff_mu / np.abs(mu)

    abs_diff_sigma = np.abs(sample_sigma - sigma)

    return {
        "n_samples": len(agents),
        "target_mu": mu,
        "sample_mu": sample_mu,
        "rel_diff_mu": rel_diff_mu,
        "target_sigma": sigma,
        "sample_sigma": sample_sigma,
        "abs_diff_sigma": abs_diff_sigma,
        "mu_matches_5pct": bool(np.all(rel_diff_mu < 0.05)),
    }


def plot_agent_distribution(agents: np.ndarray, 
                            mu: np.ndarray = MU_DEFAULT, 
                            sigma: np.ndarray = SIGMA_DEFAULT, 
                            title: str = "Engine B: Multivariate Gaussian Behavioral Sampling", 
                            ax=None, 
                            save_path: Optional[str] = None):
    """
    Scatter plot of sampled agent parameters overlaid with theoretical 1-sigma and 2-sigma confidence ellipses.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    # Scatter plot of sampled agents
    ax.scatter(agents[:, 0], agents[:, 1], alpha=0.3, color='#3a86ef', edgecolors='none', s=15, label='Sampled Agents')

    # Mean marker
    ax.scatter(mu[0], mu[1], color='#e63946', marker='X', s=120, label=f'Mean μ = ({mu[0]:.2f}, {mu[1]:.2f})', zorder=5)

    # Compute Eigendecomposition of Covariance Matrix Sigma for Confidence Ellipses
    vals, vecs = np.linalg.eigh(sigma)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # 1-sigma and 2-sigma ellipses (1-sigma ~ 39.3% in 2D, 2-sigma ~ 86.5% in 2D)
    for n_std, color, label in zip([1, 2], ['#e63946', '#ffb703'], ['1σ Confidence Ellipse', '2σ Confidence Ellipse']):
        width, height = 2 * n_std * np.sqrt(vals)
        ell = Ellipse(xy=mu, width=width, height=height, angle=angle,
                      edgecolor=color, facecolor='none', linestyle='--', linewidth=2.0, label=label, zorder=4)
        ax.add_patch(ell)

    ax.set_xlabel("Walking Speed (m/s)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Reaction Time (s)", fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax
