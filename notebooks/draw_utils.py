# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import altair as alt
from altair.vegalite.v4.schema.core import Axis, Legend 

def draw_curve(df, y_scale=[0, 1], fig_title="", y_title="Y Title", x_key="timecode", y_key="em:Q", height=800, width=1150, x_scale=[0, 100], color_dom=None, color_range=None, orient="top-right"):

    df = df[(df["timecode"] <= x_scale[1]) & (df["timecode"] >= x_scale[0])]
    x = alt.X(x_key, type="ordinal", title="", axis=alt.Axis(tickCount=10, grid=False)) 
    
    if color_dom and color_range:
        color=alt.Color('prefix:N', scale=alt.Scale(domain=color_dom, range=color_range), sort=color_dom)  
        color_wo_lengend = alt.Color('prefix:N', scale=alt.Scale(domain=color_dom, range=color_range), sort=color_dom, legend=None)  
        shape=alt.Shape('prefix:N', sort=color_dom)
        shape_wo_legend = shape=alt.Shape('prefix:N', sort=color_dom, legend=None)
    else:
        color=alt.Color('prefix:N', )  
        color_wo_lengend = alt.Color('prefix:N', legend=None)  
        shape=alt.Shape('prefix:N', )
        shape_wo_legend = shape=alt.Shape('prefix:N', legend=None)
    
    
    
     # scale=alt.Scale(range=['cross', 'circle', 'square', 'triangle-right', 'diamond'])
    y=alt.Y(y_key, stack=None, title="", scale=alt.Scale(domain=y_scale), axis=alt.Axis(tickCount=10, grid=False))

    points = alt.Chart(df).mark_point(opacity=0.8, filled=True, size=350).encode(x=x, y=y, shape=shape ,color=color).properties(title=fig_title) 

    lines = alt.Chart(df).mark_line(point=False).encode(x=x, y=y, color=color).properties(title=fig_title)
    fig = alt.layer(points, lines).resolve_scale(color="independent", shape="independent")
    # fig = points

    
    fig = fig.properties(width=width, height=height).configure_axis(
        labelFontSize=30,
        titleFontSize=30, 
    ).configure_view(stroke="black", strokeWidth=3)
    if orient != "none":
        fig = fig.configure_legend(titleFontSize=0, labelFontSize=30, symbolSize=300, orient=orient, strokeColor='gray',
            fillColor='#EEEEEE',
            padding=5, 
            cornerRadius=3,).configure_title(
            fontSize=30,
            font='Courier',
            anchor='middle',
            orient="top", align="center",
            color='black'
        )
        
    return fig


def draw_stacked_bars(df, y_scale=[0, 1], fig_title="", y_title="Y Title", x_key="timecode", y_key="em:Q", height=800, width=1150, x_scale=[0, 100], bin_width=10, color_dom=None, color_range=None):
    
    if color_dom and color_range:
        color=alt.Color('prefix:N', scale=alt.Scale(domain=color_dom, range=color_range))  
    else:
        color=alt.Color('prefix:N')  

    dom = ['Europe', 'Japan', 'USA']
    rng = ['red', 'green', 'black']

    fig = alt.Chart(df).mark_bar().encode(x=alt.X(x_key, title="Time Step", axis=alt.Axis(tickMinStep=10, tickOffset=0, tickWidth=5,), scale=alt.Scale(domain=x_scale)), 
                                            y=alt.Y(y_key, title=y_title, scale=alt.Scale(domain=y_scale)), 
                                            color=color,).properties(title=fig_title)


    fig = alt.layer(fig).resolve_scale()
    fig = fig.properties(width=width, height=height).configure_title(fontSize=50,
    ).configure_bar(binSpacing=0, width=bin_width).configure_axis(
        labelFontSize=25,
        titleFontSize=25,  
    ).configure_legend(titleFontSize=0, labelFontSize=30, orient='top-left', strokeColor='gray',
        fillColor='#EEEEEE',
        padding=5,
        cornerRadius=3,).configure_title(
        fontSize=30,
        font='Courier',
        anchor='middle',
        orient="top", align="center",
        color='black'
    )
    return fig




def draw_grouped_bars(df, y_scale=[0, 1], fig_title="", y_title="Y Title", x_key="timecode", group_key="", y_key="em:Q", height=800, width=1150, x_scale=[0, 100], bin_width=10, color_dom=None, color_range=None, orient = "none"):
    
    if color_dom and color_range:
        color=alt.Color(x_key, scale=alt.Scale(domain=color_dom, range=color_range), sort=color_dom, legend=None)  
    else:
        color=alt.Color(x_key)  
 
    bars = alt.Chart(df).mark_bar(clip=True).encode(x=alt.X(x_key, title="CL Method", sort=color_dom), 
                                            y=alt.Y(y_key, title=y_title, scale=alt.Scale(domain=y_scale), axis=alt.Axis(grid=False)), 
                                            color=color).properties(title=fig_title)


    text = bars.mark_text(
            align='left',
            baseline='middle',
            dx=3  # Nudges text to right so it doesn't appear on top of the bar
        ).encode(
            text=y_key
        )
        
    fig = bars
    # fig = alt.layer(fig).resolve_scale()
    fig = fig.properties(width=width, height=height).configure_title(fontSize=0,
    ).configure_bar(binSpacing=0, width=bin_width).configure_axis(
        labelFontSize=10,
        titleFontSize=10,  
    )
    if orient != "none":
        fig = fig.configure_legend(titleFontSize=0, labelFontSize=30, orient='top-left', strokeColor='gray',
            fillColor='#EEEEEE',
            padding=5,
            cornerRadius=3,)
    fig = fig.configure_title(
            fontSize=10,
            font='Courier',
            anchor='middle',
            orient="top", align="center",
            color='black'
        )
    return fig

