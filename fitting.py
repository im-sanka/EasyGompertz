import streamlit as st
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(layout="wide")

# Define the Gompertz function
def gompertz(x, a, b, c):
    return a * np.exp(-b * np.exp(-c * x))

def plot_gompertz(datasets, x_axis_name, y_axis_name, fit_names, initial_params):
    fig = go.Figure()

    for idx, data in enumerate(datasets):
        xdata = np.array(data['x'])
        ydata = np.array(data['y'])
        dataset_name = data['dataset_name']
        fit_name = fit_names[idx]

        if len(xdata) == 0 or len(ydata) == 0:
            st.warning(f"{dataset_name} is empty. Please provide data or upload a file.")
            continue

        p0 = initial_params

        try:
            popt, _ = curve_fit(gompertz, xdata, ydata, p0=p0)
        except RuntimeError as e:
            st.warning(f"Optimal parameters not found for {dataset_name}. Try adjusting initial parameter guesses.")
            continue

        xplot = np.linspace(0, max(xdata), 1000)
        yplot = gompertz(xplot, *popt)

        fig.add_trace(go.Scatter(x=xdata, y=ydata, mode='markers', name=dataset_name))
        fig.add_trace(go.Scatter(x=xplot, y=yplot, mode='lines', name=f'{fit_name}'))

    fig.update_layout(title="Gompertz Plots", xaxis_title=x_axis_name, yaxis_title=y_axis_name)
    return fig

def heuristic_initial_guess(data):
    """Provide a heuristic for initial guess based on data."""
    a = max(data['y'])
    b = 1.0
    # Check if data length is sufficient
    if len(data['x']) > 1:
        slope = (data['y'][1] - data['y'][0]) / (data['x'][1] - data['x'][0])
        c = slope / (a - data['y'][0]) if a != data['y'][0] else 0.1
    else:
        c = 0.1
    return (a, b, c)

st.title('Gompertz Fitting Dashboard')
st.info("**IMPORTANT**: In this platform, you can either add your data manually or upload a .csv/ .xlsx data that has two columns only."
        "  \nIf you upload the file, the first column needs to represent X axis and the second column will represent Y axis.")
datasets = []
fit_names = []

# Number of datasets input
col1, col2, col3 = st.columns(3)
num_datasets = col1.number_input("Number of datasets", 1, 5, 1)
with st.expander("**About the Gompertz Function**"):
    st.markdown("""
    The Gompertz function is a type of mathematical model for a time series and is named after Benjamin Gompertz. It is commonly used in biology to describe the growth of organisms, the number of cases in epidemics, and the decay of technological products' popularity.

    The Gompertz function is defined as:
    f(i)=a x exp(−b x exp(−c x i))

    Where:
    - \( a \) is the upper asymptote (can be the maximum of the y-values)
    - \( b \) relates to the displacement along the x-axis (time axis and can be set to 1 initially)
    - \( c \) defines the growth rate (can be approximated from the slope of the linear part of the curve)

    Choosing appropriate initial guesses for these parameters is essential for successful curve fitting. If the fitting algorithm struggles to converge, adjusting these initial guesses can often help.
    """)
col_a, col_b, col_c = st.columns(3)
use_heuristic = st.checkbox("Use heuristic for initial guess based on uploaded data")


if use_heuristic and len(datasets) > 0:
    a_guess, b_guess, c_guess = heuristic_initial_guess(datasets[0])  # Assuming heuristic from first dataset
    a_guess = st.number_input("Initial guess for a (Heuristic applied)", 0.0, 10.0, a_guess)
    b_guess = st.number_input("Initial guess for b (Heuristic applied)", 0.0, 10.0, b_guess)
    c_guess = st.number_input("Initial guess for c (Heuristic applied)", 0.0, 10.0, c_guess)
else:
    a_guess = col_a.number_input("Initial guess for a", 0.0, 10.0, 1.0)
    b_guess = col_b.number_input("Initial guess for b", 0.0, 10.0, 0.1)
    c_guess = col_c.number_input("Initial guess for c", 0.0, 10.0, 0.1)

for i in range(num_datasets):
    st.markdown(f"### Dataset {i+1}")
    col1,col2 = st.columns(2)
    dataset_name = col1.text_input(f"Name for Dataset {i+1}", f"Dataset {i+1}")
    fit_name = col2.text_input(f"Name for Fit {i+1} in Legend", f"Gompertz Fit {i+1}")
    fit_names.append(fit_name)

    uploaded_file = st.file_uploader(f"Or, upload CSV or XLSX file for dataset {i+1}", type=["csv", "xlsx"])

    if uploaded_file:
        # Resetting the file position to the start
        uploaded_file.seek(0)

        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Get the first and second columns regardless of their names
        x_values = data.iloc[:, 0].tolist()
        y_values = data.iloc[:, 1].tolist()

        datasets.append({'x': x_values, 'y': y_values, 'dataset_name': dataset_name})
    else:
        x_values = col1.text_input(f'Enter x values for {dataset_name} (comma-separated with no space e.g. 0.1,0.2,0.3):')
        y_values = col2.text_input(f'Enter y values for {dataset_name} (comma-separated with no space e.g. 0.05,0.15,0.30):')

        try:
            x_values = [float(x) for x in x_values.split(',')] if x_values else []
            y_values = [float(y) for y in y_values.split(',')] if y_values else []
        except ValueError:
            st.error(f"Error in processing input for {dataset_name}. Ensure correct format.")
            continue

        if len(x_values) != len(y_values):
            st.error(f"Length mismatch between x and y values for {dataset_name}")
            continue

        datasets.append({'x': x_values, 'y': y_values, 'dataset_name': dataset_name})

# Inputs for x and y axis names
st.subheader("Renaming the axis:")
col1,col2 = st.columns(2)
x_axis_name = col1.text_input("Name for X-axis", "X-axis label")
y_axis_name = col2.text_input("Name for Y-axis", "Y-axis label")

if st.button('Plot and fit!'):
    initial_params = (a_guess, b_guess, c_guess)
    fig = plot_gompertz(datasets, x_axis_name, y_axis_name, fit_names, initial_params)
    st.plotly_chart(fig, use_container_width=True)
