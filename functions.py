import streamlit as st
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import pandas as pd

# Define the Gompertz function
def gompertz(x, a, b, c):
    return a * np.exp(-b * np.exp(-c * x))

def plot_gompertz(datasets, x_axis_name, y_axis_name, fit_names, initial_params):
    fig = go.Figure()

    for idx, data in enumerate(datasets):
        try:
            xdata = np.array(data['x'])
            ydata = np.array(data['y'])
            dataset_name = data['dataset_name']
            fit_name = fit_names[idx]
            color = data.get('color', 'gray')

            if 'std_dev' in data:
                std_dev = np.array(data['std_dev'])
            else:
                std_dev = None

            if len(xdata) == 0 or len(ydata) == 0:
                raise ValueError(f"{dataset_name} is empty. Please provide data or upload a file.")

            p0 = initial_params

            popt, _ = curve_fit(gompertz, xdata, ydata, p0=p0, sigma=std_dev)

            xplot = np.linspace(0, max(xdata), 1000)
            yplot = gompertz(xplot, *popt)

            # Adding deviation bars if available
            if std_dev is not None:
                fig.add_trace(go.Scatter(x=xdata, y=ydata, mode='markers', error_y=dict(type='data', array=std_dev, visible=True), name=dataset_name, marker=dict(color=color)))
            else:
                fig.add_trace(go.Scatter(x=xdata, y=ydata, mode='markers', name=dataset_name, marker=dict(color=color)))

            fig.add_trace(go.Scatter(x=xplot, y=yplot, mode='lines', name=f'{fit_name}', line=dict(color=color)))

        except ValueError as e:
            st.error(f"Error processing data for {dataset_name}: {str(e)}")
            continue  # skip this dataset and move to the next one
        except RuntimeError as e:
            st.warning(f"Optimal parameters not found for {dataset_name}. Try adjusting initial parameter guesses.")
            continue

    fig.update_layout(title={'text': "Gompertz Plots",'font': {'size': 24}},
                      xaxis={'title': x_axis_name, 'titlefont': {'size': 20,'color': 'black'},
                             'tickfont': {'size': 18,'color': 'black'},'showline':True},
                      yaxis={'title': y_axis_name,'titlefont': {'size': 20,'color': 'black'},
                             'tickfont': {'size': 18,'color': 'black'},'showline':True,'showgrid':False},
                      legend={'font': {'size': 16}})
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