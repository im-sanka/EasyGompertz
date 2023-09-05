import streamlit as st
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import pandas as pd
from functions import gompertz, plot_gompertz, heuristic_initial_guess

st.set_page_config(page_title="EasyGompertz", page_icon=":chart_with_downwards_trend:",layout="wide")

st.title('Gompertz Fitting Dashboard')
datasets = []
fit_names = []

# Number of datasets input
col1, col2 = st.columns(2)
num_datasets = col1.number_input("Number of datasets", 1, 5, 1)
col2.info("**IMPORTANT**: In this platform, you can either add your data manually or upload a .csv/ .xlsx data."
        " If you upload the file, the first column needs to represent X axis and the second column will represent Y axis and the third column is the standard deviation.")

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

available_colors = ["lime", "cyan", "red", "teal","orange", "#ffc857", "#119da4"]


for i in range(num_datasets):
    st.markdown(f"### Dataset {i+1}")
    col1, col2, col3 = st.columns(3)  # Added an extra column for color selection

    dataset_name = col1.text_input(f"Name for Dataset {i+1}", f"Dataset {i+1}")
    fit_name = col2.text_input(f"Name for Fit {i+1} in Legend", f"Gompertz Fit {i+1}")

    dataset_color = col3.selectbox(f"Color for {dataset_name}", available_colors, index=i%len(available_colors))  # Defaulting to a different color for each dataset initially
    fit_names.append(fit_name)

    uploaded_file = st.file_uploader(f"Or, upload CSV or XLSX file for dataset {i+1}", type=["csv", "xlsx"])

    if uploaded_file:
        # Resetting the file position to the start
        uploaded_file.seek(0)

        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Check number of columns to determine if standard deviations are included
        if data.shape[1] == 3:
            x_values = data.iloc[:, 0].tolist()
            y_values = data.iloc[:, 1].tolist()
            std_dev = data.iloc[:, 2].tolist()
            datasets.append({'x': x_values, 'y': y_values, 'std_dev': std_dev, 'dataset_name': dataset_name, 'color': dataset_color.lower()})
        elif data.shape[1] == 2:
            x_values = data.iloc[:, 0].tolist()
            y_values = data.iloc[:, 1].tolist()
            datasets.append({'x': x_values, 'y': y_values, 'dataset_name': dataset_name, 'color': dataset_color.lower()})
        else:
            st.error(f"Unexpected number of columns in uploaded file for {dataset_name}. Expecting 2 or 3 columns.")
            continue

    else:
        x_values = col1.text_input(f'Enter x values for {dataset_name} (comma-separated with no space e.g. 0.1,0.2,0.3):')
        y_values = col2.text_input(f'Enter y values for {dataset_name} (comma-separated with no space e.g. 0.05,0.15,0.30):')
        std_dev = col3.text_input(f'Enter standard deviations for {dataset_name} (optional, comma-separated with no space e.g. 0.01,0.02,0.03):', "")

        # ... [rest of the processing remains unchanged]

        if std_dev:
            try:
                std_dev = [float(s) for s in std_dev.split(',')] if std_dev else []
            except ValueError:
                st.error(f"Error in processing standard deviations for {dataset_name}. Ensure correct format.")
                continue

            if len(std_dev) != len(x_values):
                st.error(f"Length mismatch between x values and standard deviations for {dataset_name}")
                continue

            datasets.append({'x': x_values, 'y': y_values, 'std_dev': std_dev, 'dataset_name': dataset_name, 'color': dataset_color.lower()})
        else:
            datasets.append({'x': x_values, 'y': y_values, 'dataset_name': dataset_name, 'color': dataset_color.lower()})


# Inputs for x and y axis names
st.subheader("Renaming the axis:")
col1,col2 = st.columns(2)
x_axis_name = col1.text_input("Name for X-axis", "X-axis label")
y_axis_name = col2.text_input("Name for Y-axis", "Y-axis label")

if st.button('Plot and fit!'):
    try:
        initial_params = (a_guess, b_guess, c_guess)
        fig = plot_gompertz(datasets, x_axis_name, y_axis_name, fit_names, initial_params)
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("Please add the data either manually of data upload!")

