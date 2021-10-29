import altair as alt 

def draw_curve(df, y_scale=[0, 1], fig_title="", y_title="Y Title", x_key="timecode", y_key="em:Q", height=800):
    x = alt.X(x_key, type="ordinal", title="Timecode") 
    color=alt.Color('prefix:N')  
    fig = alt.Chart(df).mark_line(opacity=0.7,  point=True).encode(x=x, y=alt.Y(y_key, stack=None, title=y_title, scale=alt.Scale(domain=y_scale)), color=color).properties(title=fig_title)

    fig = alt.layer(fig).resolve_scale()
    fig = fig.properties(width=1100, height=height).configure_axis(
        labelFontSize=18,
        titleFontSize=16, 
    ).configure_legend(titleFontSize=0, labelFontSize=20, orient='top-right', strokeColor='gray',
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


def draw_stacked_bars(df, y_scale=[0, 1], fig_title="", y_title="Y Title", x_key="timecode", y_key="em:Q", height=800):
    x = alt.X(x_key, type="ordinal", title="Timecode", ) # axis=alt.Axis(tickCount=10)
    color=alt.Color('prefix:N')  
    fig = alt.Chart(df).mark_bar().encode(x=x, y=alt.Y(y_key, title=y_title, scale=alt.Scale(domain=y_scale)), color=color,).properties(title=fig_title)

    fig = alt.layer(fig).resolve_scale()
    fig = fig.properties(width=1100, height=height).configure_axis(
        labelFontSize=10,
        titleFontSize=16, 
        tickCount=10,
    ).configure_legend(titleFontSize=0, labelFontSize=20, orient='top-right', strokeColor='gray',
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

