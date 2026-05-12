import streamlit as st
import numpy as np

st.set_page_config(layout="wide")

st.title("MC Error Lower Bound Explorer")
st.caption("Interactive verification of theoretical Monte Carlo lower bounds of error in $L^p$ approximation of classes containg $\\tanh$ neural networks.")


def num_input(name, min_val, default, inf_possible=True, max_val=None, step=1, auto_possible=True):
    if name not in st.session_state:
        st.session_state[name] = default
        if inf_possible:
            st.session_state[f"{name}_inf"] = False
        elif auto_possible:
            st.session_state[f"{name}_auto"] = True  # ← NEW

    col1, col2 = st.sidebar.columns([3,1.5])

    # --- format ---
    if name == "$\\varepsilon_{p}$":
        fmt = "%.2e"
    elif isinstance(step, int):
        fmt = "%d"
    else:
        fmt = "%.2f"

    is_inf = st.session_state.get(f"{name}_inf", False)
    is_auto = st.session_state.get(f"{name}_auto", False)

    # --- enforce automatic value ---
    if is_auto:
        st.session_state[name] = default

    with col1:
        val = st.number_input(
            name,
            min_value=min_val,
            max_value=max_val,
            step=step,
            key=name,
            disabled=is_inf or is_auto,   # ← lock when auto
            format=fmt
        )

    # --- right column: ∞ OR auto ---
    with col2:
        if inf_possible:
            st.checkbox("∞", key=f"{name}_inf")
        elif auto_possible:
            st.checkbox("auto", key=f"{name}_auto")

    return np.inf if (inf_possible and is_inf) else val


def pretty(v):
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        if isinstance(v, float):
            return f"{v:.4g}"   # or ".3f", ".2e", etc.
        return str(v)


# ---- Computation ----
def a_comp(q,B,a):
    return B**(1-2/q)*a

def constant_before_m(B, a, q, p, s, c_B_a):
    term1 = np.sqrt((a_comp(q,B,a))**2 - 1) / (4 * B**(1 - 2/q))
    term2 = (c_B_a / (2**(1 + 2/s) * np.sqrt(s)))**(s/p)
    return term1 * term2

def c_a(B,q,a):
    B_tanh = B **(-1/q) * np.tanh(a/2)
    return np.tanh(B_tanh) / (20 * np.cosh(B_tanh)**2)

def c_B_a(B,q,a):
    B_tanh = B **(-1/q) * (a - np.tanh(a/2))
    return 3/5 * np.tanh(B_tanh) / B_tanh

def alpha(tilde_a):
    if tilde_a <= 1:
        return 0
    alpha = 1 / (np.sqrt(tilde_a ** 2 - 4 * tilde_a**2/ (tilde_a + 1)**2 * (np.arccosh(np.sqrt(tilde_a)))**2) + 1)
    return alpha

def j_assump(q,B,a, tilde_a, alpha_a):
    if tilde_a <= 1:
        return 0
    nom = 5 * B ** (3/q) * np.arccosh(np.sqrt(tilde_a)) * alpha_a
    cosh_a = (np.cosh(2*np.arccosh(np.sqrt(tilde_a))*alpha_a))**2
    den = a ** 2 * (np.tanh(2*B**(-1/q)*np.tanh(a/2))) * (np.tanh(B**(-1/q)*(a - np.tanh(a/2))))**2

    final = np.log(nom / den)/ np.log(tilde_a / cosh_a)
    return final

def pi(a, alpha_a):
    x = np.sqrt(a) + np.sqrt(a-1)
    return x ** (4*(1-alpha_a))

def k_assump(e_p,a, tilde_a, alpha_a):
    if tilde_a <= 1:
        return 0
    p = pi(a, alpha_a)
    nom = 4*a*tilde_a / (e_p * p * (1+1/p)**2)
    den = (np.cosh(tilde_a * (p-1)/(p+1)))**2 / tilde_a
    return 3 + np.log(nom) / np.log(den)

def s_assump(m_max, c_a, q, B, a, alpha_a, tilded_a, j):
    cosh_a = (np.cosh(2*np.arccosh(np.sqrt(tilded_a)) * alpha_a))**2
    x = tilded_a / cosh_a
    y = c_a * a ** 2 * (a - np.tanh(a/2)) ** 2
    z = y / (16 * B ** (5/q) * np.arccosh(np.sqrt(tilded_a)) * alpha_a)
    return 2*np.log(4*m_max) / (j * np.log(x) + np.log(z))

# --- Assumptions ---
st.sidebar.markdown("## Parameters")

if st.sidebar.button("Reset parameters", type="primary"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# Nondependt num_inputs
p = num_input("$p$", 1, 4, auto_possible=False)
q = num_input("$q$", 1, 4, auto_possible=False)
d = num_input("$d$", 1, 15, inf_possible=False, auto_possible=False)
B = num_input("$B$", 1, 45, inf_possible=False, auto_possible=False)
a = num_input("$a$", 0.0, 2.0, step=0.01, inf_possible=False, auto_possible=False)

# Precomputed variables
tilde_a = a_comp(q,B,a)
alpha_a = alpha(tilde_a)

st.markdown("### 1. Structural assumptions")
st.divider()

a_formula = r"B^{1-\frac2q}a"
ass_on_B = r"\qquad \text{and} \qquad B \ge 2d"

st.latex(rf"\tilde a = {a_formula} = {round(tilde_a,2)} > 1, \quad B \ge 2d")

if tilde_a <= 1:
    st.error(r"Violation: $\tilde a \le 1$")
elif B < 2*d:
    st.error(r"Violation: $B < 2d$")
else:
    st.success("All structural assumptions satisfied.")

    st.markdown("### 2. Layer allocation")
    st.divider()


    e_p = num_input("$\\varepsilon_{p}$", 0.0, 1e-16, step = 1e-16, inf_possible=False, auto_possible=False)

    j_ass = j_assump(q, B, a, tilde_a, alpha_a)
    j_min = max(1, int(np.ceil(j_ass)))
    k_ass = k_assump(e_p, a, tilde_a, alpha_a)
    k_min = max(3, int(np.ceil(k_ass)))
    L_min = k_min+j_min + 3
    L = num_input("$L$", min_val=L_min, default=max(L_min, 12), inf_possible=False, auto_possible=False)

    k = num_input("$k$", k_min, k_min, max_val=L-3, inf_possible=False, auto_possible=True)
    j_max = L - 3 - k
    j = num_input("$j$", j_min, j_max, max_val=j_max, inf_possible=False, auto_possible=True)

    k_assump_formula = r"3 + \frac{\ln\!\left( \frac{4a \tilde a}{\varepsilon_p \pi(\tilde a)\left(1+\pi(\tilde a)^{-1}\right)^2} \right)}{\ln \!\left( \frac{\cosh^2\!\left(\tilde a\frac{\pi(\tilde a)-1}{\pi(\tilde a)+1}\right)}{\tilde a} \right)} \le k"

    st.write("Assumption on $k \\ge 3$, which controls the tail suppresion:")
    st.latex(rf"{round(k_ass,2)} = {k_assump_formula} \, .")

    st.write("Assumptions on $j$, which increases $m$:")

    j_assump_formula = r"\frac{\ln\left( \frac{5 \, B^{\frac3q}\operatorname{arccosh}\left(\sqrt{\tilde a}\right)\,\rho(\tilde a)}{a^2\tanh(2B^{-1/q}\tanh(\frac a2))\tanh^2\left[B^{-1/q}(a - \tanh(\frac a2))\right]} \right)}{\ln\left(  \frac{\tilde a}{\cosh^2\left[2\operatorname{arccosh}(\sqrt{\tilde a})\rho(\tilde a)\right]} \right)}\leq j"
    st.latex(rf"{round(j_ass, 2)} = {j_assump_formula} \, .")
    st.write("Hence:")

    L_formula = r"L-3 = k + j \implies L \ge 3 + j_{\min} + k_{\min}"
    st.latex(rf"{L_formula} = {L_min} \, .")

    st.markdown("##### Total number of weight parameters:")
    P = (L-2)*B**2 + (L+d)*B + 1

    st.latex(rf"P = B^2(L-2) + B(L+d) + 1 = {P}")

    m_max = num_input(r"$m_{\\max}$", 1, 100000, inf_possible=False, auto_possible=False)

    if P > m_max:
        st.warning("Number of parameters exceeds sample budget ($P > m_{\max}$).")

    const_c_a = c_a(B,q,a)
    const_c_B_a = c_B_a(B,q,a)

    s_ass = s_assump(m_max, const_c_a, q, B, a, alpha_a, tilde_a, j)


    if s_ass < 0:
        st.error("Unable to get $s$ positive, adjust the inputs")
    elif s_ass > d:
        st.error("$s$ has to be greater than $d$ in order to satisfy the results for all $m \\leq m_{\\max}$, adjust the inputs")
    else:
        s_formula = r"d \;\ge\; s \;\ge\;\frac{2\ln(4m_{\max})}{j\cdot\ln\!\Big( \frac{\tilde a}{\cosh^2\!\left[2\operatorname{arccosh}(\sqrt{\tilde a})\rho(\tilde a)\right]} \Big)   +  \ln\!\left(\frac{c_a\,a^2(a-\tanh(a/2))^2}{16\,B^{5/q}\operatorname{arccosh}(\sqrt{\tilde a})\rho(\tilde a)}\right)}"
        st.markdown("### 3. Dimension constraint")
        st.divider()

        st.latex(rf"{s_formula} = {round(s_ass, 2)} \, .")

        s_min = max(1, int(np.ceil(s_ass)))
        s = num_input("$s$", s_min, s_min, max_val=d, inf_possible=False, auto_possible=True)

        # ---- Output ----
        st.markdown("### 4. Final bound")
        st.divider()
        
        c_a_formula = r"c_a = \frac{\tanh\left(B^{-1/q}\tanh\left(\frac a2\right)\right)}{20\cosh^2\left(B^{-1/q}\tanh\left(\frac a2\right)\right)}"
        c_B_a_formula = r"\qquad \qquad \bar c_{B,a} = \frac{3}{5} \cdot \frac{\tanh\left(B^{-1/q}\left(a-\tanh\left(\frac a2\right)\right)\right)}{B^{-1/q}\left(a-\tanh\left(\frac a2\right)\right)}"

        #st.latex(rf"{c_a_formula} = {const_c_a:.2e}, {c_B_a_formula} = {const_c_B_a:.2e}")

        c_Ba = c_B_a(B, q, a)
        final_const = constant_before_m(B, a, q, p, s, c_Ba)
        m_form = r"\cdot m^{-\frac1p}"
        error_formula = r"\operatorname{err}_m^{MC}\!\left(U, L^p([0,1]^d)\right)\;\ge\;"
        worst_error_formula = r"\operatorname{err}_{m_{\max}}^{MC}\!\left(U, L^p([0,1]^d)\right)\;\ge\;"
        lower_bound_formula = r"\frac{\sqrt{B^{2-\frac{4}{q}}a^2-1}}{4B^{1-\frac{2}{q}}}\cdot\left(\frac{\bar c_{B,a}}{2^{1+\frac{2}{s}}\sqrt{s}}\right)^{\frac{s}{p}}m^{-\frac{1}{p}}"
        mantissa, exp = f"{final_const:.2e}".split("e")
        exp = int(exp)
        worst_lower_bound = final_const * m_max ** (-1/p)

        if final_const < e_p:
            st.metric("Lower bound constant", f"0")
            st.latex(rf"{error_formula} 0")
            st.error("Lower bound is smaller than machine precision $\\varepsilon_{p}$ for every $m$")
        else:
            st.metric("Lower bound constant", f"{final_const:.2e}")
            st.latex(rf"{error_formula}{lower_bound_formula} = {mantissa} \cdot 10^{{{exp}}} {m_form}")
            st.write("The worst case lower bound:")
            st.latex(rf"{worst_error_formula} {worst_lower_bound:.2e}")
            if worst_lower_bound < e_p:
                st.error("Lower bound is smaller than machine precision for some $m$")