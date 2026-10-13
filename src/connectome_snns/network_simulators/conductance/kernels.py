"""Triton kernels for the conductance-based LIF dynamics.

They implement exactly the time step of ``dynamics.py`` (the readable reference), fused
so that a neuron's state never leaves registers within a chunk:

* ``scan_forward`` / ``scan_backward`` integrate a chunk whose synaptic drive is known
  in advance (feedforward networks, and the backward pass of every network) — one
  launch per chunk, the time loop inside the kernel, BPTT as the matching reverse scan.
* ``compact_spikes`` + ``recurrent_step`` advance a recurrent network by one step.
  Spikes are sparse (~1% of neurons per ms), so instead of a dense ``spikes @ W``
  each step gathers only the weight rows of the neurons that fired.

Layouts (all contiguous float32 unless noted):

    v (B, N)    g (B, 2, S, N) — the networks expose it as a (B, N, 2, S) view
    drive A (K, B, T, N)    spikes (B, T, N)
    saved for backward: v (B, T, N), summed conductances G (B, T, N, S)
    per-pathway tables: scale (K, N), made (K, SP)
    per-synapse tables, padded with zeros to SP, the next power of two > S:
        jump (2, SP), decay (2, SP), E_syn (SP), g_clip (SP)

``g``, the saved ``G`` and the recurrent weights ``W`` may instead be float16 (the
network's ``storage_dtype``): loads convert them to float32, stores round back.

``K`` pathways (one per source cell type and input side), ``S`` synapse types. A padded
synapse column has zero jump, decay and clip, so it stays at zero and adds no current.
"""

import triton
import triton.language as tl


@triton.jit
def _spike(v, theta, noise, surr_scale, SAMPLE: tl.constexpr):
    """Spikes from the voltage entering the step (Heaviside, or Bernoulli escape noise)."""
    if SAMPLE:
        p = tl.sigmoid(surr_scale * (v - theta))
        return (noise < p).to(tl.float32)
    return ((v - theta) > 0).to(tl.float32)


@triton.jit
def _membrane(v, s, gr, gd, U_reset, E_L, C_m, beta, E_syn, g_clip, dt):
    """Synaptic and leak currents, and the voltage after the step (with reset)."""
    G = gr + gd
    Gc = tl.minimum(tl.maximum(G, 0.0), g_clip[None, :])
    I = Gc * (v[:, None] - E_syn[None, :])
    I_leak = (v - E_L) * (1 - beta) * C_m / dt
    v_next = (v - (tl.sum(I, axis=1) + I_leak) * dt / C_m) * (1 - s) + U_reset * s
    return G, I, I_leak, v_next


@triton.jit
def _track(
    track_v, track_I, track_g, b, n, t, T, N, j, live,
    v_next, I, I_leak, gr, gd, track_b,
    S: tl.constexpr,
):  # fmt: skip
    """Write tracked variables for the tracked trial (or every trial if track_b < 0)."""
    tracked = live & ((track_b < 0) | (b == track_b))
    bt = tl.where(track_b < 0, b, 0)
    row = (bt * T + t) * N + n
    tl.store(track_v + row, v_next, mask=tracked)
    # The leak current goes in column S, which the padding guarantees is free.
    I_all = tl.where(j[None, :] == S, I_leak[:, None], I)
    tl.store(
        track_I + row[:, None] * (S + 1) + j[None, :],
        I_all,
        mask=tracked[:, None] & (j[None, :] <= S),
    )
    jm = tracked[:, None] & (j[None, :] < S)
    tl.store(track_g + row[:, None] * (2 * S) + j[None, :], gr, mask=jm)
    tl.store(track_g + row[:, None] * (2 * S) + S + j[None, :], gd, mask=jm)


@triton.jit
def scan_forward(
    A, scale, made, jump, decay, E_syn, g_clip,
    theta, U_reset, E_L, C_m, beta,
    v_state, g_state, spikes, prob, noise,
    v_saved, G_saved, track_v, track_I, track_g,
    dt_ptr, surr_ptr, track_b, t0, TA, B, T, N,
    K: tl.constexpr, S: tl.constexpr, SP: tl.constexpr, BLOCK: tl.constexpr,
    SAVE: tl.constexpr, TRACK: tl.constexpr, PROB: tl.constexpr, SAMPLE: tl.constexpr,
):  # fmt: skip
    """Integrate steps ``t0 .. t0 + TA`` of a ``T``-step chunk.

    ``A`` holds the pathway activations of just these ``TA`` steps, so a long chunk can
    be fed in blocks; outputs are indexed by the step within the whole chunk.
    """
    dt = tl.load(dt_ptr)  # scalars read on the device: no host sync to pass them
    surr_scale = tl.load(surr_ptr)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)  # flat (trial, neuron)
    live = idx < B * N
    b = idx // N
    n = idx % N
    j = tl.arange(0, SP)
    sm = live[:, None] & (j[None, :] < S)

    theta_n = tl.load(theta + n, mask=live)
    U_reset_n = tl.load(U_reset + n, mask=live)
    E_L_n = tl.load(E_L + n, mask=live)
    C_m_n = tl.load(C_m + n, mask=live, other=1.0)
    beta_n = tl.load(beta + n, mask=live)
    E = tl.load(E_syn + j)
    clip = tl.load(g_clip + j)
    jump_r, jump_d = tl.load(jump + j), tl.load(jump + SP + j)
    decay_r, decay_d = tl.load(decay + j), tl.load(decay + SP + j)

    v = tl.load(v_state + idx, mask=live)
    g_row = g_state + (b * (2 * S * N) + n)[:, None] + j[None, :] * N
    gr = tl.load(g_row, mask=sm, other=0.0).to(tl.float32)
    gd = tl.load(g_row + S * N, mask=sm, other=0.0).to(tl.float32)

    for i in range(TA):
        t = t0 + i
        out = (b * T + t) * N + n
        if SAMPLE:
            u = tl.load(noise + out, mask=live, other=1.0)
        else:
            u = 0.0
        s = _spike(v, theta_n, u, surr_scale, SAMPLE)
        tl.store(spikes + out, s, mask=live)
        if PROB:
            tl.store(prob + out, tl.sigmoid(surr_scale * (v - theta_n)), mask=live)
        G, I, I_leak, v_next = _membrane(
            v, s, gr, gd, U_reset_n, E_L_n, C_m_n, beta_n, E, clip, dt
        )
        if SAVE:
            tl.store(v_saved + out, v, mask=live)
            tl.store(G_saved + out[:, None] * S + j[None, :], G, mask=sm)

        gr = gr * decay_r[None, :]
        gd = gd * decay_d[None, :]
        for k in tl.static_range(K):
            a = tl.load(
                A + k * (B * TA * N) + (b * TA + i) * N + n, mask=live, other=0.0
            )
            sc = tl.load(scale + k * N + n, mask=live, other=0.0)
            mk = tl.load(made + k * SP + j)
            gr += a[:, None] * (jump_r[None, :] * sc[:, None] * mk[None, :])
            gd += a[:, None] * (jump_d[None, :] * sc[:, None] * mk[None, :])
        v = v_next
        if TRACK:
            _track(
                track_v, track_I, track_g, b, n, t, T, N, j, live,
                v, I, I_leak, gr, gd, track_b, S,
            )  # fmt: skip

    tl.store(v_state + idx, v, mask=live)
    tl.store(g_row, gr, mask=sm)
    tl.store(g_row + S * N, gd, mask=sm)


@triton.jit
def scan_backward(
    grad_spikes, grad_prob, spikes, v_saved, G_saved, A, scale, made, jump, decay,
    E_syn, g_clip, theta, C_m, beta, grad_A, grad_scale,
    dt_ptr, surr_ptr, B, T, N,
    K: tl.constexpr, KP: tl.constexpr, S: tl.constexpr, SP: tl.constexpr,
    BLOCK: tl.constexpr, WANT_A: tl.constexpr, WANT_SCALE: tl.constexpr,
    PROB: tl.constexpr, SAMPLE: tl.constexpr,
):  # fmt: skip
    """Backpropagate through ``T`` steps: the adjoint of ``scan_forward``.

    Carries the adjoints of the voltage and of each conductance component backwards in
    time. Writes ``grad_A`` (gradient w.r.t. the pathway activations) and/or the
    per-neuron partial sums of the gradient w.r.t. each pathway's ``scale``.
    """
    dt = tl.load(dt_ptr)
    surr_scale = tl.load(surr_ptr)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < B * N
    b = idx // N
    n = idx % N
    j = tl.arange(0, SP)
    sm = live[:, None] & (j[None, :] < S)
    kk = tl.arange(0, KP)

    theta_n = tl.load(theta + n, mask=live)
    C_m_n = tl.load(C_m + n, mask=live, other=1.0)
    beta_n = tl.load(beta + n, mask=live)
    E = tl.load(E_syn + j)
    clip = tl.load(g_clip + j)
    jump_r, jump_d = tl.load(jump + j), tl.load(jump + SP + j)
    decay_r, decay_d = tl.load(decay + j), tl.load(decay + SP + j)

    dv = tl.zeros((BLOCK,), dtype=tl.float32)  # dL/dv entering the next step
    lr = tl.zeros((BLOCK, SP), dtype=tl.float32)  # dL/dg (rise) entering the next step
    ld = tl.zeros((BLOCK, SP), dtype=tl.float32)  # dL/dg (decay)
    acc = tl.zeros((BLOCK, KP), dtype=tl.float32)  # sum_t dL/dscale_k

    for i in range(T):
        t = T - 1 - i
        out = (b * T + t) * N + n
        v = tl.load(v_saved + out, mask=live)
        s = tl.load(spikes + out, mask=live, other=0.0)
        G = tl.load(G_saved + out[:, None] * S + j[None, :], mask=sm, other=0.0)
        G = G.to(tl.float32)

        # The injections of step t enter the state that step t + 1 sees.
        for k in tl.static_range(K):
            mk = tl.load(made + k * SP + j)
            q = tl.sum(
                (jump_r[None, :] * lr + jump_d[None, :] * ld) * mk[None, :], axis=1
            )
            if WANT_A:
                sc = tl.load(scale + k * N + n, mask=live, other=0.0)
                tl.store(grad_A + k * (B * T * N) + out, q * sc, mask=live)
            if WANT_SCALE:
                a = tl.load(A + k * (B * T * N) + out, mask=live, other=0.0)
                acc += tl.where(kk[None, :] == k, (q * a)[:, None], 0.0)

        # Membrane update: v_next = (v - (sum_j I_j + I_leak) dt / C) (1 - s) + U s.
        Gc = tl.minimum(tl.maximum(G, 0.0), clip[None, :])
        gx = dv * (1 - s)
        gy = -gx * dt / C_m_n
        clamp = tl.where(G < clip[None, :], 1.0, tl.where(G == clip[None, :], 0.5, 0.0))
        clamp = tl.where(G >= 0.0, clamp, 0.0)
        dG = gy[:, None] * (v[:, None] - E[None, :]) * clamp
        dv = gx + gy * (tl.sum(Gc, axis=1) + (1 - beta_n) * C_m_n / dt)

        # Spike (surrogate) and probability outputs of step t.
        x = v - theta_n
        gs = tl.load(grad_spikes + out, mask=live, other=0.0)
        if SAMPLE:
            p = tl.sigmoid(surr_scale * x)
            dv += gs * surr_scale * p * (1.0 - p)
        else:
            dv += gs / ((surr_scale * tl.abs(x) + 1.0) * (surr_scale * tl.abs(x) + 1.0))
        if PROB:
            p = tl.sigmoid(surr_scale * x)
            dv += (
                tl.load(grad_prob + out, mask=live, other=0.0)
                * surr_scale
                * p
                * (1.0 - p)
            )

        # Conductances decay into the next step and feed the currents of this one.
        lr = lr * decay_r[None, :] + dG
        ld = ld * decay_d[None, :] + dG

    if WANT_SCALE:
        km = live[:, None] & (kk[None, :] < K)
        tl.store(grad_scale + kk[None, :] * (B * N) + idx[:, None], acc, mask=km)


@triton.jit
def compact_spikes(
    v, theta, noise, rows, row_start, lists, counts,
    surr_ptr, t, B, T, N, L,
    BLOCK: tl.constexpr, SAMPLE: tl.constexpr,
):  # fmt: skip
    """List, in order, the rows of recurrent pathway ``k`` whose neuron fires now.

    One program per (trial, pathway). ``rows`` holds every recurrent pathway's source
    neurons back to back, ``row_start`` (K + 1) where each pathway's begin.
    """
    surr_scale = tl.load(surr_ptr)
    b = tl.program_id(0)
    k = tl.program_id(1)
    start = tl.load(row_start + k)
    n_rows = tl.load(row_start + k + 1) - start
    count = 0
    for base in range(0, n_rows, BLOCK):
        r = base + tl.arange(0, BLOCK)
        m = r < n_rows
        neuron = tl.load(rows + start + r, mask=m, other=0)
        vv = tl.load(v + b * N + neuron, mask=m, other=0.0)
        th = tl.load(theta + neuron, mask=m, other=0.0)
        if SAMPLE:
            u = tl.load(noise + (b * T + t) * N + neuron, mask=m, other=1.0)
        else:
            u = 0.0
        fired = (_spike(vv, th, u, surr_scale, SAMPLE) > 0) & m
        position = count + tl.cumsum(fired.to(tl.int32), axis=0) - 1
        tl.store(lists + (k * B + b) * L + position, r, mask=fired)
        count += tl.sum(fired.to(tl.int32), axis=0)
    tl.store(counts + k * B + b, count)


@triton.jit
def recurrent_step(
    A_ff, W, w_start, lists, counts, scale, made, jump, decay, E_syn, g_clip,
    theta, U_reset, E_L, C_m, beta,
    v_state, g_state, spikes, noise,
    v_saved, G_saved, A_saved, track_v, track_I, track_g,
    dt_ptr, surr_ptr, track_b, t, t_ff, T, T_ff, B, N, L,
    K_REC: tl.constexpr, K_FF: tl.constexpr, S: tl.constexpr, SP: tl.constexpr,
    BLOCK: tl.constexpr, ROWS: tl.constexpr, SAVE: tl.constexpr, TRACK: tl.constexpr,
    SAMPLE: tl.constexpr,
):  # fmt: skip
    """Advance a recurrent network one step, for one trial and a block of neurons.

    Recurrent activations are gathered from the weight rows listed by
    ``compact_spikes``; feedforward ones are read from ``A_ff`` at ``t_ff``.
    """
    dt = tl.load(dt_ptr)
    surr_scale = tl.load(surr_ptr)
    b = tl.program_id(0)
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = n < N
    idx = b * N + n
    j = tl.arange(0, SP)
    sm = live[:, None] & (j[None, :] < S)

    theta_n = tl.load(theta + n, mask=live)
    E = tl.load(E_syn + j)
    clip = tl.load(g_clip + j)

    v = tl.load(v_state + idx, mask=live)
    g_row = g_state + (b * (2 * S * N) + n)[:, None] + j[None, :] * N
    gr = tl.load(g_row, mask=sm, other=0.0).to(tl.float32)
    gd = tl.load(g_row + S * N, mask=sm, other=0.0).to(tl.float32)

    out = (b * T + t) * N + n
    if SAMPLE:
        u = tl.load(noise + out, mask=live, other=1.0)
    else:
        u = 0.0
    s = _spike(v, theta_n, u, surr_scale, SAMPLE)
    tl.store(spikes + out, s, mask=live)
    G, I, I_leak, v_next = _membrane(
        v, s,
        gr, gd,
        tl.load(U_reset + n, mask=live),
        tl.load(E_L + n, mask=live),
        tl.load(C_m + n, mask=live, other=1.0),
        tl.load(beta + n, mask=live),
        E, clip, dt,
    )  # fmt: skip
    if SAVE:
        tl.store(v_saved + out, v, mask=live)
        tl.store(G_saved + out[:, None] * S + j[None, :], G, mask=sm)

    gr = gr * tl.load(decay + j)[None, :]
    gd = gd * tl.load(decay + SP + j)[None, :]
    jump_r, jump_d = tl.load(jump + j), tl.load(jump + SP + j)
    for k in tl.static_range(K_REC + K_FF):
        if k < K_REC:
            # Sum the weight rows of the neurons that fired, ROWS rows per load.
            a = tl.zeros((BLOCK,), dtype=tl.float32)
            first = tl.load(w_start + k)
            listed = lists + (k * B + b) * L
            fired = tl.load(counts + k * B + b)
            for m in range(0, fired, ROWS):
                m_ = m + tl.arange(0, ROWS)
                row = first + tl.load(listed + m_, mask=m_ < fired, other=0)
                w = tl.load(
                    W + row[:, None] * N + n[None, :],
                    mask=(m_ < fired)[:, None] & live[None, :],
                    other=0.0,
                )
                a += tl.sum(w.to(tl.float32), axis=0)
            if SAVE:
                tl.store(A_saved + k * (B * T * N) + out, a, mask=live)
        else:
            a = tl.load(
                A_ff + (k - K_REC) * (B * T_ff * N) + (b * T_ff + t_ff) * N + n,
                mask=live,
                other=0.0,
            )
        sc = tl.load(scale + k * N + n, mask=live, other=0.0)
        mk = tl.load(made + k * SP + j)
        gr += a[:, None] * (jump_r[None, :] * sc[:, None] * mk[None, :])
        gd += a[:, None] * (jump_d[None, :] * sc[:, None] * mk[None, :])

    tl.store(v_state + idx, v_next, mask=live)
    tl.store(g_row, gr, mask=sm)
    tl.store(g_row + S * N, gd, mask=sm)
    if TRACK:
        _track(
            track_v, track_I, track_g, b, n, t, T, N, j, live,
            v_next, I, I_leak, gr, gd, track_b, S,
        )  # fmt: skip
