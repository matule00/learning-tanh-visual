import streamlit as st
import numpy as np

st.set_page_config(layout="wide")
st.title("MC Error Lower Bound")
st.write(
    "Computational illustration of the resulted bound."
)

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

def constant(B, a, q, p, s):
    term1 = np.sqrt(B**(2 - 4/q) * a**2 - 1) / (4 * B**(1 - 2/q))
    c_bar = c_B_a(B, a, q)
    term2 = (c_bar / (2**(1 + 2/s) * np.sqrt(s)))**(s/p)
    return term1 * term2

def c_a(B,q,a):
    B_tanh = B **(-1/q) * np.tanh(a/2)
    return np.tanh(B_tanh) / (20 * np.cosh(B_tanh)**2)

def c_B_a(B,q,a):
    B_tanh = B **(-1/q) * (a - np.tanh(a/2))
    return np.sqrt(5/12) * np.tanh(B_tanh) / B_tanh

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
    den = 2 * a * (np.tanh(a - np.tanh(a/2)))**3 * cosh_a

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
st.sidebar.header("Parameters")

# Nondependt num_inputs
p = num_input("p", 1, 4)
q = num_input("q", 1, 4)
B = num_input("B", 1, 45, inf_possible=False)
a = num_input("a", 0.0, 2.0, step=0.01, inf_possible=False)

# Precomputed variables
tilde_a = a_comp(q,B,a)
alpha_a = alpha(tilde_a)

st.subheader("Assumptions")

st.write("Necessary:")
ass_on_a = r"1< B^{1-\frac2q}a"
st.latex(rf"{ass_on_a} = {round(tilde_a,2)}")

if tilde_a <= 1:
    st.error("Condition not satisfied")
else:
    st.success("Condition satisfied")

    d = num_input("d", 1, 15, inf_possible=False)
    e_p = num_input("e_p", 0.0, 10e-16, step = 10e-32, inf_possible=False)

    L_ass = L_assump(q, B, a, tilde_a, alpha_a)
    k_ass = k_assump(e_p, a, tilde_a, alpha_a)
    L_min = max(int(np.ceil(L_ass)), 6, int(np.ceil(3+k_ass)))
    L = num_input("L", min_val=L_min, default=max(L_min, 12), inf_possible=False)

    j_ass = L - 2 - k_ass

    k = num_input("k", int(np.ceil(k_ass)), int(np.ceil(k_ass)), inf_possible=False, max_val=L-3)
    j = num_input("j", int(np.ceil(k_ass)), int(np.ceil(k_ass)), inf_possible=False, value_override=L-2-k)

    k_assump_formula = r"k \;\ge\; 3 + \frac{\ln\!\left( \frac{4a \tilde a}{\varepsilon_p \pi(\tilde a)\left(1+\pi(\tilde a)^{-1}\right)^2} \right)}{\ln \!\left( \frac{\cosh^2\!\left(\tilde a\frac{\pi(\tilde a)-1}{\pi(\tilde a)+1}\right)}{\tilde a} \right)}"
    j_assump_formula = r"\qquad j \leq L-2-k"

    st.write("Assumption on k to push the tail error below machine precision and the remaining number of layers j to increase m::")
    st.latex(rf"{k_assump_formula} = {k_ass:.2}, {j_assump_formula} = {j_ass:.2}")

    st.write("Assumptions on L:")
    st.latex(r"""
    L \ge 6,\quad 
    L-2 = k + j 
    \implies 
    L \ge 3 + k_{\min}.
    """)

    st.write("and")

    L_assump_formula = r"\frac{2\ln\left( \frac{21 B^{1+1/q}\operatorname{arccosh}\left(\sqrt{\tilde a}\right)\,\alpha(\tilde a)}{2a\tanh^3\left[a - \tanh(a/2)\right]\cosh^2\left[2\operatorname{arccosh}(\sqrt{\tilde a})\alpha(\tilde a)\right]} \right)}{\ln\left(  \frac{\tilde a}{\cosh^2\left[2\operatorname{arccosh}(\sqrt{\tilde a})\alpha(\tilde a)\right]} \right)} +2 \leq L"
    st.latex(rf"{L_ass:.2} = {L_assump_formula}")
    st.write("Hence:")
    st.latex(rf"L \ge \min(6, {round(L_ass, 2)}, {int(np.ceil(3+k_ass))}) = {L_min}")


    m_max = num_input(r"m_max", 1, 100000, inf_possible=False)

    const_c_a = c_a(B,q,a)
    const_c_B_a = c_B_a(B,q,a)

    s_ass = s_assump(m_max, const_c_a, q, B, a, alpha_a, tilde_a, j)
    s_formula = r"d \;\ge\; s \;\ge\;\frac{2\ln(4m_{\max})}{j\,\ln\!\Big( \frac{\tilde a}{\cosh^2\!\left[2\operatorname{arccosh}(\sqrt{\tilde a})\alpha(\tilde a)\right]} \Big)+\ln\!\left(\frac{c_a\,a(a-\tanh(a/2))^2\cosh^2\!\left[2\operatorname{arccosh}(\sqrt{\tilde a})\alpha(\tilde a)\right]}{16\,B^{1+3/q}\arccosh(\sqrt{\tilde a})\,\alpha(\tilde a)}\right)}"
    st.latex(rf"{s_formula} = {round(s_ass, 2)}")


    if s_ass < 1:
        st.error("Unable to get s positive, adjust the inputs")
    elif s_ass > d:
        st.error("s is greater than d, adjust the inputs")
    else:
        st.success("s positive")

        s = num_input("s", int(np.ceil(s_ass)), int(np.ceil(s_ass)), inf_possible=False, max_val=d)



        # ---- Compute ----
        C = constant(B, a, q, p, s)
        bound = C * m_max**(-1/p)

        # ---- Output ----
        st.subheader("Result")

        c_a_formula = r"c_a = \frac{\tanh\left(B^{-1/q}\tanh\left(\frac a2\right)\right)}{20\cosh^2\left(B^{-1/q}\tanh\left(\frac a2\right)\right)}"
        c_B_a_formula = r"\qquad \qquad \bar c_{B,a} = \sqrt{\frac{5}{12}} \cdot \frac{\tanh\left(B^{-1/q}\left(a-\tanh\left(\frac a2\right)\right)\right)}{B^{-1/q}\left(a-\tanh\left(\frac a2\right)\right)}"

        st.latex(rf"{c_a_formula} = {const_c_a:.2e}, {c_B_a_formula} = {const_c_B_a:.2e}")

        st.metric("Constant C", f"{C:.4e}")
        st.metric("Lower bound", f"{bound:.4e}")


        # ---- Table ----

        st.subheader("Details")
        st.table({
            "parameter": ["p", "q", "B", "a", "L", "k", "j", "s", "m_max"],
            "value": [
                pretty(p),
                pretty(q),
                pretty(B),
                pretty(L),
                pretty(k),
                pretty(j),
                pretty(a),
                pretty(s),
                pretty(m_max),
            ]
        })

        # ---- Optional small plot ----
        if st.checkbox("Show m ↦ bound plot"):
            ms = np.linspace(1, m_max, 100)
            vals = C * ms**(-1/p)
            st.line_chart(vals)