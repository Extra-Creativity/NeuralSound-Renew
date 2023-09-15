import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def get_figure(coords, data):
    '''
        coords: (N, 3)
        data: (N, M)
    '''

    # Create figure
    fig = go.Figure()
    # Add traces, one for each slider step
    for i in range(data.shape[1]):
        color_data = data[:, i]
        fig.add_trace(
            go.Scatter3d(
                visible=False,
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    # set color to an array/list of desired values
                    color=color_data,
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.8

                )
            )
        )

    # Make 10th trace visible
    fig.data[0].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": "Switched to feature: " + str(i)}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Feature: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    return fig