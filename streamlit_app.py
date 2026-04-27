import streamlit as st
import numpy as np

st.title("MC Error Lower Bound")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

def slider_with_input(name, min_val, max_val, default, step=1):
    if name not in st.session_state:
        st.session_state[name] = default
        st.session_state[f"{name}_slider"] = default
        st.session_state[f"{name}_input"] = default

    def sync_from_slider():
        val = st.session_state[f"{name}_slider"]
        st.session_state[name] = val
        st.session_state[f"{name}_input"] = val

    def sync_from_input():
        val = st.session_state[f"{name}_input"]
        st.session_state[name] = val
        st.session_state[f"{name}_slider"] = val

    col1, col2 = st.sidebar.columns([2,1])

    with col1:
        st.slider(name, min_val, max(max_val, st.session_state[f"{name}_input"]),
                  key=f"{name}_slider",
                  step=step,
                  on_change=sync_from_slider)

    with col2:
        st.number_input("",
                        min_val,
                        key=f"{name}_input",
                        step=step,
                        on_change=sync_from_input)

    return st.session_state[name]

def slider_with_inf(name, min_val, max_val, default, step=1):
    if name not in st.session_state:
        st.session_state[name] = default
        st.session_state[f"{name}_inf"] = False

    col1, col2, col3 = st.sidebar.columns([2,1,1])

    with col1:
        val = st.slider(name, min_val, max_val, st.session_state[name], step=step)

    with col2:
        val_input = st.number_input("", min_val, max_val, value=val, step=step)

    with col3:
        is_inf = st.checkbox("∞", key=f"{name}_inf")

    # logic
    if is_inf:
        return np.inf
    else:
        st.session_state[name] = val_input if val_input != val else val
        return st.session_state[name]

# ---- Inputs ----
st.sidebar.header("Parameters")

p = slider_with_inf("p", 1, 20, 5)
q = slider_with_input("q", 1, 20, 5)
d = slider_with_input("d", 1, 100, 20)
e = slider_with_input("e", 0.0, 1.0, 10e-16, step = 10e-32)
B = slider_with_input("B", 1, 100, 20)
a = slider_with_input("a", 0.0, 20.0, 2.0, step=0.1)
s = slider_with_input("s", 1, 5, 2)
m = slider_with_input("m", 10, 10000, 100)


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