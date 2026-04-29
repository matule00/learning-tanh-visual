import streamlit as st
import numpy as np

st.set_page_config(layout="wide")

st.title("MC Error Lower Bound Explorer")
st.caption("Interactive verification of theoretical Monte Carlo lower bounds of error in $L^p$ approximation of classes containg $\\tanh$ neural networks.")


def num_input(name, min_val, default, inf_possible=True, max_val=None, step=1, value_override=None):
    if name not in st.session_state:
        st.session_state[name] = default
        if inf_possible:
            st.session_state[f"{name}_inf"] = False

    col1, col2 = st.sidebar.columns([3,1])

    # --- format ---
    if name == "e_p":
        fmt = "%.2e"
    elif isinstance(step, int):
        fmt = "%d"
    else:
        fmt = "%.2f"

    # --- inf state ---
    is_inf = st.session_state.get(f"{name}_inf", False)

    with col1:
        if max_val is not None:
            val = st.number_input(name, 
                              min_value=min_val,
                              max_value=max_val,
                              step=step, 
                              key=name, 
                              disabled=is_inf, 
                              format=fmt 
                              )
        else:
            if value_override is not None:
                val = st.number_input(
                    name,
                    value=int(value_override),
                    step=1,
                    format="%d",
                    disabled=True
                )
                return value_override
            
            val = st.number_input(name, 
                                min_value=min_val,
                                step=step, 
                                key=name, 
                                disabled=is_inf, 
                                format=fmt 
                                )

    # --- show ∞ only if allowed ---
    if inf_possible:
        with col2:
            st.checkbox("∞", key=f"{name}_inf")

    if inf_possible and is_inf:
        return np.inf
    else:
        return val

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

def L_assump(q,B,a, tilde_a, alpha_a):
    if tilde_a <= 1:
        return 0
    nom = 21 * B ** (1+1/q) * np.arccosh(np.sqrt(tilde_a)) * alpha_a
    cosh_a = (np.cosh(2*np.arccosh(np.sqrt(tilde_a))*alpha_a))**2
    den = 2 * a * (np.tanh(B**(-1/q)*(a - np.tanh(a/2))))**3 * cosh_a

    final = 2*np.log(nom / den)/ np.log(tilde_a / cosh_a) + 2
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
    y = c_a * a * (a - np.tanh(a/2)) ** 2 * cosh_a
    z = y / (16 * B ** (1+3/q) * np.arccosh(np.sqrt(tilded_a)) * alpha_a)
    return 2*np.log(4*m_max) / (j * np.log(x) + np.log(z))

# --- Assumptions ---
st.sidebar.markdown("## Parameters")

if st.sidebar.button("Reset parameters", type="primary"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# Nondependt num_inputs
p = num_input("$p$", 1, 4)
q = num_input("$q$", 1, 4)
d = num_input("$d$", 1, 15, inf_possible=False)
B = num_input("$B$", 1, 45, inf_possible=False)
a = num_input("$a$", 0.0, 2.0, step=0.01, inf_possible=False)

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


    e_p = num_input("$\\varepsilon_{p}$", 0.0, 10e-16, step = 10e-32, inf_possible=False)

    L_ass = L_assump(q, B, a, tilde_a, alpha_a)
    k_ass = k_assump(e_p, a, tilde_a, alpha_a)
    k_min = max(3, int(np.ceil(k_ass)))
    L_min = max(int(np.ceil(L_ass)), 6, k_min+3)
    L = num_input("$L$", min_val=L_min, default=max(L_min, 12), inf_possible=False)

    j_ass = L - 2 - k_ass
    j_min = L - 2 - k_min

    k = num_input("$k$", k_min, k_min, inf_possible=False, max_val=L-3)
    j = num_input("$j$", j_min, j_min, inf_possible=False, value_override=L-2-k)

    k_assump_formula = r"k \;\ge\; 3 + \frac{\ln\!\left( \frac{4a \tilde a}{\varepsilon_p \pi(\tilde a)\left(1+\pi(\tilde a)^{-1}\right)^2} \right)}{\ln \!\left( \frac{\cosh^2\!\left(\tilde a\frac{\pi(\tilde a)-1}{\pi(\tilde a)+1}\right)}{\tilde a} \right)}"
    j_assump_formula = r"\qquad j \leq L-2-k"

    st.write("$k \\ge 3$ controls the tail suppresion and $j$ increases $m$:")
    st.latex(rf"{k_assump_formula} = {k_ass:.2}, {j_assump_formula} = {j_ass:.2}")

    st.write("Assumptions on $L \\geq 6$:")
    st.latex(r"""
    L-2 = k + j 
    \implies 
    L \ge 3 + k_{\min}.
    """)

    st.write("and")

    L_assump_formula = r"\frac{2\ln\left( \frac{21 B^{1+1/q}\operatorname{arccosh}\left(\sqrt{\tilde a}\right)\,\alpha(\tilde a)}{2a\tanh^3\left[B^{-1/q}(a - \tanh(a/2))\right]\cosh^2\left[2\operatorname{arccosh}(\sqrt{\tilde a})\alpha(\tilde a)\right]} \right)}{\ln\left(  \frac{\tilde a}{\cosh^2\left[2\operatorname{arccosh}(\sqrt{\tilde a})\alpha(\tilde a)\right]} \right)} +2 \leq L"
    st.latex(rf"{L_ass:.2} = {L_assump_formula}")
    st.write("Hence:")
    st.latex(rf"L \ge \min(\,{round(L_ass, 2)}, \,3+{k_ass:.2}\,) = {L_min}")

    st.markdown("##### Total number of weight parameters:")
    P = (L-2)*B**2 + (L+d)*B + 1

    st.latex(rf"P = B^2(L-2) + B(L+d) + 1 = {P}")

    m_max = num_input(r"$m_{\\max}$", 1, 100000, inf_possible=False)

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
        s_formula = r"d \;\ge\; s \;\ge\;\frac{2\ln(4m_{\max})}{j\,\ln\!\Big( \frac{\tilde a}{\cosh^2\!\left[2\operatorname{arccosh}(\sqrt{\tilde a})\alpha(\tilde a)\right]} \Big)+\ln\!\left(\frac{c_a\,a(a-\tanh(a/2))^2\cosh^2\!\left[2\operatorname{arccosh}(\sqrt{\tilde a}) \alpha(\tilde a)\right]}{16\,B^{1+3/q}\operatorname{arccosh}(\sqrt{\tilde a})\alpha(\tilde a)}\right)}"
        st.markdown("### 3. Dimension constraint")
        st.divider()

        st.latex(rf"{s_formula} = {round(s_ass, 2)}")

        s_min = max(1, int(np.ceil(s_ass)))
        s = num_input("$s$", s_min, s_min, inf_possible=False, max_val=d)

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