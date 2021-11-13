import altair as alt 

def draw_curve(df, y_scale=[0, 1], fig_title="", y_title="Y Title", x_key="timecode", y_key="em:Q", height=800):
    x = alt.X(x_key, type="ordinal", title="Timecode") 
    color=alt.Color('prefix:N')  
    fig = alt.Chart(df).mark_line(opacity=0.7,  point=True).encode(x=x, y=alt.Y(y_key, stack=None, title=y_title, scale=alt.Scale(domain=y_scale)), color=color).properties(title=fig_title)

    fig = alt.layer(fig).resolve_scale()
    fig = fig.properties(width=1100, height=height).configure_axis(
        labelFontSize=18,
        titleFontSize=16, 
    ).configure_legend(titleFontSize=0, labelFontSize=20, orient='top-left', strokeColor='gray',
        fillColor='#EEEEEE',
        padding=5,
        cornerRadius=3,).configure_title(
        fontSize=20,
        font='Courier',
        anchor='middle',
        orient="top", align="center",
        color='black'
    )
    return fig


def draw_stacked_bars(df, y_scale=[0, 1], fig_title="", y_title="Y Title", x_key="timecode", y_key="em:Q", height=800, width=1150, x_scale=[0, 100]):
    
    color=alt.Color('prefix:N')  
    fig = alt.Chart(df).mark_bar().encode(x=alt.X(x_key, title="Time Step", axis=alt.Axis(tickMinStep=10, tickOffset=0, tickWidth=5,), scale=alt.Scale(domain=x_scale)), 
                                            y=alt.Y(y_key, title=y_title, scale=alt.Scale(domain=y_scale)), 
                                            color=color,).properties(title=fig_title)

    fig = alt.layer(fig).resolve_scale()
    fig = fig.properties(width=width, height=height).configure_title(fontSize=50,
    ).configure_bar(binSpacing=0, width=15).configure_axis(
        labelFontSize=25,
        titleFontSize=25,  
    ).configure_legend(titleFontSize=0, labelFontSize=25, orient='top-left', strokeColor='gray',
        fillColor='#EEEEEE',
        padding=5,
        cornerRadius=3,).configure_title(
        fontSize=25,
        font='Courier',
        anchor='middle',
        orient="top", align="center",
        color='black'
    )
    return fig

