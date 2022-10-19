# Built on top of the original implementation at https://github.com/md-mohaiminul/ViS4mer
#
# Modifications by Copyright 2022 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from einops import rearrange
from scipy import special as ss
import numpy as np
from opt_einsum import contract
from pykeops.torch import Genred

_conj = lambda x: torch.cat([x, x.conj()], dim=-1)


def _broadcast_dims(*tensors):
    max_dim = max([len(tensor.shape) for tensor in tensors])
    tensors = [tensor.view((1,)*(max_dim-len(tensor.shape))+tensor.shape) for tensor in tensors]
    return tensors


def cauchy_conj(v, z, w, num=2, denom=2):
    """
    Pykeops version
    Args:
        v (torch.tensor): (..., N, N)
        z (torch.tensor): (..., N, L)
        w (torch.tensor): (..., N, L)
        num (int): (..., N, L)
        denom (int): (..., N, L)
    """
    if num == 1:
        expr_num = 'z * ComplexReal(v) - Real2Complex(ComplexReal(v)*ComplexReal(w) + ComplexImag(v)*ComplexImag(w))'
    elif num == 2:
        expr_num = 'z * ComplexReal(v) - Real2Complex(Sum(v * w))'
    else:
        raise NotImplementedError

    if denom == 1:
        expr_denom = 'ComplexMult(z-Real2Complex(ComplexReal(w)),' \
                     ' z-Real2Complex(ComplexReal(w))) + Real2Complex(Square(ComplexImag(w)))'
    elif denom == 2:
        expr_denom = 'ComplexMult(z-w, z-Conj(w))'
    else:
        raise NotImplementedError

    cauchy_mult = Genred(
        f'ComplexDivide({expr_num}, {expr_denom})',
        # expr_num,
        # expr_denom,
        [
            'v = Vj(2)',
            'z = Vi(2)',
            'w = Vj(2)',
        ],
        reduction_op='Sum',
        axis=1,
        dtype='float32' if v.dtype == torch.cfloat else 'float64',
    )

    v, z, w = _broadcast_dims(v, z, w)
    v = torch.view_as_real(v)
    z = torch.view_as_real(z)
    w = torch.view_as_real(w)

    r = 2*cauchy_mult(v, z, w, backend="CPU")
    return torch.view_as_complex(r)


def power(l, a, v=None):
    """
    Compute a^l and the scan sum_i a^i v_i
    Args:
        l (int): power
        a (torch.tensor): (..., N, N)
        v (torch.tensor): (..., N, L)
    """

    i = torch.eye(a.shape[-1]).to(a)  # dtype=a.dtype, device=a.device

    powers = [a]
    ll = 1
    while True:
        if l % 2 == 1:
            i = powers[-1] @ i
        l //= 2
        if l == 0:
            break
        ll *= 2
        powers.append(powers[-1] @ powers[-1])

    if v is None:
        return i

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible
    # and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.size(-1) - ll
    v_ = powers.pop() @ v[..., ll:]
    v = v[..., :ll]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return i, v.squeeze(-1)


def krylov(l, a, b, c=None, return_power=False):
    """
    Compute the Krylov matrix (b, ab, a^2b, ...) using the squaring trick.
    If return_power=True, return A^{L-1} as well
    Args:
        a (torch.tensor): torch.tensor
        b (torch.tensor): torch.tensor
        c (torch.tensor): torch.tensor
        return_power (bool): return_power
    """
    # TODO There is an edge case if l=1 where output doesn't get broadcasted, which might be an issue
    #  if caller is expecting broadcasting semantics... can deal with it if it arises

    x = b.unsqueeze(-1)  # (..., N, 1)
    a_ = a

    al = None
    if return_power:
        al = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
        _l = l-1

    done = l == 1
    # loop invariant: _L represents how many indices left to compute
    while not done:
        if return_power:
            if _l % 2 == 1:
                al = a_ @ al
            _l //= 2

        # Save memory on last iteration
        ll = x.shape[-1]
        if l - ll <= ll:
            done = True
            _x = x[..., :l-ll]
        else:
            _x = x

        _x = a_ @ _x
        x = torch.cat([x, _x], dim=-1)  # there might be a more efficient way of ordering axes
        if not done:
            a_ = a_ @ a_

    assert x.shape[-1] == l

    if c is not None:
        x = torch.einsum('...nl, ...n -> ...l', x, c)
    x = x.contiguous()  # WOW!!
    if return_power:
        return x, al
    else:
        return x


def transition(measure, n, **measure_args):
    """
    Return a, bb transition matrices for different measures
    Args:
        measure (str): the type of measure
            legt - Legendre (translated)
            legs - Legendre (scaled)
            glagt - generalized Laguerre (translated)
            lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
        n (int): dimension
    """
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        a = np.eye(n) / 2 - np.tril(np.ones((n, n)))
        bb = b * np.ones((n, 1))
    elif measure == 'tlagt':
        # beta = 1 corresponds to no tilt
        b = measure_args.get('beta', 1.0)
        a = (1.-b)/2 * np.eye(n) - np.tril(np.ones((n, n)))
        bb = b * np.ones((n, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    elif measure == 'glagt':
        alpha = measure_args.get('alpha', 0.0)
        beta = measure_args.get('beta', 0.01)
        a = -np.eye(n) * (1 + beta) / 2 - np.tril(np.ones((n, n)), -1)
        b = ss.binom(alpha + np.arange(n), np.arange(n))[:, None]

        l = np.exp(.5 * (ss.gammaln(np.arange(n)+alpha+1) - ss.gammaln(np.arange(n)+1)))
        a = (1./l[:, None]) * a * l[None, :]
        bb = (1./l[:, None]) * b * np.exp(-.5 * ss.gammaln(1-alpha)) * beta**((1-alpha)/2)
    # Legendre (translated)
    elif measure == 'legt':
        q = np.arange(n, dtype=np.float64)
        r = (2*q + 1) ** .5
        j, i = np.meshgrid(q, q)
        a = r[:, None] * np.where(i < j, (-1.)**(i-j), 1) * r[None, :]
        bb = r[:, None]
        a = -a
    # LMU: equivalent to LegT up to normalization
    elif measure == 'lmu':
        q = np.arange(n, dtype=np.float64)
        r = (2*q + 1)[:, None]  # / theta
        j, i = np.meshgrid(q, q)
        a = np.where(i < j, -1, (-1.)**(i-j+1)) * r
        bb = (-1.)**q[:, None] * r
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(n, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        m = -(np.where(row >= col, r, 0) - np.diag(q))
        t = np.sqrt(np.diag(2 * q + 1))
        a = t @ m @ np.linalg.inv(t)
        bb = np.diag(t)[:, None]
        bb = bb.copy()  # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    else:
        raise NotImplementedError

    return a, bb


def rank_correction(measure, n, rank=1, dtype=torch.float):
    """
    Return low-rank matrix L such that A + L is normal
    Args:
        measure (str): the type of measure
            legt - Legendre (translated)
            legs - Legendre (scaled)
            glagt - generalized Laguerre (translated)
            lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
        n (int): dimension
        rank (int): rank
        dtype (str): data type
    """

    if measure == 'legs':
        assert rank >= 1
        p = torch.sqrt(.5+torch.arange(n, dtype=dtype)).unsqueeze(0)  # (1 n)
    elif measure == 'legt':
        assert rank >= 2
        p = torch.sqrt(1+2*torch.arange(n, dtype=dtype))  # (n)
        p0 = p.clone()
        p0[0::2] = 0.
        p1 = p.clone()
        p1[1::2] = 0.
        p = torch.stack([p0, p1], dim=0)  # (2 n)
    elif measure == 'lagt':
        assert rank >= 1
        p = .5**.5 * torch.ones(1, n, dtype=dtype)
    else:
        raise NotImplementedError

    d = p.size(0)
    if rank > d:
        p = torch.stack([p, torch.zeros(n, dtype=dtype).repeat(rank-d, d)], dim=0)  # (rank n)
    return p


def nplr(measure, n, rank=1, dtype=torch.float):
    """
    Return w, p, q, v, b such that
    (w - p q^*, b) is unitarily equivalent to the original HiPPO a, b by the matrix v
    i.e. a = v[w - p q^*]v^*, b = v b
    Args:
        measure (str): the type of measure
            legt - Legendre (translated)
            legs - Legendre (scaled)
            glagt - generalized Laguerre (translated)
            lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
        n (int): dimension
        rank (int): rank
        dtype (str): data type
    """
    a, b = transition(measure, n)
    a = torch.as_tensor(a, dtype=dtype)  # (n, n)
    b = torch.as_tensor(b, dtype=dtype)[:, 0]  # (n,)

    p = rank_correction(measure, n, rank=rank, dtype=dtype)
    ap = a + torch.sum(p.unsqueeze(-2)*p.unsqueeze(-1), dim=-3)
    w, v = torch.linalg.eig(ap)  # (..., n) (..., n, n)
    # v w v^{-1} = A

    # Only keep one of the conjugate pairs
    w = w[..., 0::2].contiguous()
    v = v[..., 0::2].contiguous()

    v_inv = v.conj().transpose(-1, -2)

    b = contract('ij, j -> i', v_inv, b.to(v))  # v^* b
    p = contract('ij, ...j -> ...i', v_inv, p.to(v))  # v^* p

    return w, p, p, b, v
