import streamlit as st
import numpy as np

st.title("MC Error Lower Bound")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

def number_with_inf(name, min_val, default, step=1):
    if name not in st.session_state:
        st.session_state[name] = default
        st.session_state[f"{name}_inf"] = False

    col1, col2 = st.sidebar.columns([3,1])

    with col1:
        val = st.number_input(
            name,
            min_value=min_val,
            value=st.session_state[name],
            step=step
        )

    with col2:
        is_inf = st.checkbox("∞", key=f"{name}_inf")

    if is_inf:
        return np.inf
    else:
        st.session_state[name] = val
        return val

# ---- Inputs ----
st.sidebar.header("Parameters")

p = number_with_inf("p", 1, 4)
q = number_with_inf("q", 1, 4)
d = number_with_inf("d", 1, 5)
e = number_with_inf("e", 0.0, 10e-16, step = 10e-32)
B = number_with_inf("B", 1, 40)
a = number_with_inf("a", 0.0, 2.0, step=0.1)
s = number_with_inf("s", 1, 2)
m = number_with_inf("m", 1, 10e5)


# ---- Constants ----
def bar_c(B, a, q):
    x = B**(-1/q) * (a - np.tanh(a/2))
    return np.sqrt(5/12) * np.tanh(x) / x

def constant(B, a, q, p, s):
    term1 = np.sqrt(B**(2 - 4/q) * a**2 - 1) / (4 * B**(1 - 2/q))
    c_bar = bar_c(B, a, q)
    term2 = (c_bar / (2**(1 + 2/s) * np.sqrt(s)))**(s/p)
    return term1 * term2

# ---- Compute ----
C = constant(B, a, q, p, s)
bound = C * m**(-1/p)

# ---- Output ----
st.subheader("Result")

st.metric("Constant C", f"{C:.4e}")
st.metric("Lower bound", f"{bound:.4e}")

# ---- Table ----
st.subheader("Details")
st.table({
    "parameter": ["p", "q", "B", "a", "s", "m"],
    "value": [p, q, B, a, s, m]
})

# ---- Optional small plot ----
if st.checkbox("Show m ↦ bound plot"):
    ms = np.linspace(1, m, 100)
    vals = C * ms**(-1/p)
    st.line_chart(vals)