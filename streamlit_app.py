import streamlit as st
import numpy as np

st.title("MC Error Lower Bound")
st.write(
    "Computational illustration of the resulted bound."
)

def number_with_inf(name, min_val, default, inf_possible=True, step=1):
    if name not in st.session_state:
        st.session_state[name] = default
        if inf_possible:
            st.session_state[f"{name}_inf"] = False

    col1, col2 = st.sidebar.columns([3,1])

    # --- format ---
    if name == "e":
        fmt = "%.2e"
    elif isinstance(step, int):
        fmt = "%d"
    else:
        fmt = "%.2f"

    # --- inf state ---
    is_inf = st.session_state.get(f"{name}_inf", False)

    with col1:
        val = st.number_input(
            name,
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

def show_const(name, expr, value):
    st.latex(rf"{name} = {expr} = {value:.2e}")


# ---- Inputs ----
st.sidebar.header("Parameters")

p = number_with_inf("p", 1, 4)
q = number_with_inf("q", 1, 4)
d = number_with_inf("d", 1, 5, inf_possible=False)
e = number_with_inf("e", 0.0, 10e-16, step = 10e-32, inf_possible=False)
B = number_with_inf("B", 1, 40, inf_possible=False)
a = number_with_inf("a", 0.0, 2.0, step=0.01, inf_possible=False)
s = number_with_inf("s", 1, 2, inf_possible=False)
m_max = number_with_inf(r"m_max", 1, 100000, inf_possible=False)


# ---- Constants ----
def bar_c(B, a, q):
    x = B**(-1/q) * (a - np.tanh(a/2))
    return np.sqrt(5/12) * np.tanh(x) / x

def constant(B, a, q, p, s):
    term1 = np.sqrt(B**(2 - 4/q) * a**2 - 1) / (4 * B**(1 - 2/q))
    c_bar = bar_c(B, a, q)
    term2 = (c_bar / (2**(1 + 2/s) * np.sqrt(s)))**(s/p)
    return term1 * term2

def c_a(B,q,a):
    B_tanh = B **(-1/q) * np.tanh(a/2)
    return np.tanh(B_tanh) / (20 * np.cosh(B_tanh)**2)

def c_B_a(B,q,a):
    B_tanh = B **(-1/q) * (a - np.tanh(a/2))
    return np.sqrt(5/12) * np.tanh(B_tanh) / B_tanh

def alpha(q,B,a):
    bar_a = B ** (1-2/q) * a
    alpha = 1 / (np.sqrt(bar_a ** 2 - 4 * bar_a**2/ (bar_a + 1)**2 * (np.arccosh(np.sqrt(bar_a)))**2) + 1)
    return alpha

def L_assump(q,B,a):
    bar_a = B ** (1-2/q) * a
    nom = 21 * B ** (1+1/q) * np.arccosh(np.sqrt(bar_a)) * alpha(q,B,a)
    cosh_a = (np.cosh(2*np.arccosh(np.sqrt(bar_a))*alpha(q,B,a)))**2
    den = 2 * a * (np.tanh(a - np.tanh(a/2)))**3 * cosh_a

    final = 2*np.log(nom / den)/ np.log(bar_a / cosh_a) + 2
    return final

# ---- Compute ----
C = constant(B, a, q, p, s)
bound = C * m_max**(-1/p)

# ---- Output ----
st.subheader("Result")

c_a_formula = r"\frac{\tanh\left(B^{-1/q}\tanh\left(\frac a2\right)\right)}{20\cosh^2\left(B^{-1/q}\tanh\left(\frac a2\right)\right)}"
show_const("c_a", c_a_formula, c_a(B,q,a))

c_B_a_formula = r"\sqrt{\frac{5}{12}} \cdot \frac{\tanh\left(B^{-1/q}\left(a-\tanh\left(\frac a2\right)\right)\right)}{B^{-1/q}\left(a-\tanh\left(\frac a2\right)\right)}"
show_const(r"\bar{c}_{B,a}", c_B_a_formula, c_B_a(B,q,a))

L_assump_formula = r"\frac{2\ln\left( \frac{21 B^{1+\frac1q}\operatorname{arccosh}\left(\sqrt{\tilde a}\right)\,\alpha(\tilde a)}{2a\tanh^3\left[a - \tanh(a/2)\right]\cosh^2\left[2\operatorname{arccosh}(\sqrt{\tilde a})\alpha(\tilde a)\right]} \right)}{\ln\left(  \frac{\tilde a}{\cosh^2\left[2\operatorname{arccosh}(\sqrt{\tilde a})\alpha(\tilde a)\right]} \right)} +2 \leq L"
st.latex(rf"{L_assump(q,B,a):.2} = {L_assump_formula}")

st.metric("Constant C", f"{C:.4e}")
st.metric("Lower bound", f"{bound:.4e}")

# ---- Table ----

def pretty(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.4g}"   # or ".3f", ".2e", etc.
    return str(v)


st.subheader("Details")
st.table({
    "parameter": ["p", "q", "B", "a", "s", "m_max"],
    "value": [
        pretty(p),
        pretty(q),
        pretty(B),
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