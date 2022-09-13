
import jax

def gram(A):
    if jax.numpy.isrealobj(A):
        return A.T @ A
    G = mat_hermitian(A) @ A
    return G.real


def mat_hermitian(a):
    return jax.numpy.conjugate(jax.numpy.swapaxes(a, -1, -2))


def coherence_with_index(A):
    G = gram(A)
    G = jax.numpy.abs(G)
    n = G.shape[0]
    # set diagonals to 0
    G = G.at[jax.numpy.diag_indices(n)].set(0)
    index = jax.numpy.unravel_index(jax.numpy.argmax(G, axis=None), G.shape)
    max_val = G[index]
    return max_val, index


def coherence(A):
    max_val, index = coherence_with_index(A)
    return max_val