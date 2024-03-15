# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
# Add title
st.title("EDA with Streamlit")
# Define the columns you want to use 
columns_to_use = ['SalePrice', 'Living Area Sqft', 'Lot Frontage', 'Bldg Type', 'Bedroom',
                    'Total Full Baths','MS Zoning','Street', 
                    'Alley','Utilities']
# Function for loading data
# Adding data caching
@st.cache_data
def load_data():
    fpath =  "Data/ames-housing-dojo-for-ml.csv"
    df = pd.read_csv(fpath)
    df = df.set_index("PID")
    df = df[columns_to_use]
    return df

# load the data 
df = load_data()
# Display an interactive dataframe
st.header("Displaying a DataFrame")
st.dataframe(df, width=800)

# Display descriptive statistics
st.markdown('#### Descriptive Statistics')
st.dataframe(df.describe().round(2))

# Capture .info()
# Create a string buffer to capture the content
buffer = StringIO()
# Write the info into the buffer
df.info(buf=buffer)
# Retrieve the content from the buffer
summary_info = buffer.getvalue()
# Use Streamlit to display the info
st.markdown("#### Summary Info")
st.text(summary_info)

# We could display the output series as a dataframe
st.markdown("#### Null Values as dataframe")
nulls =df.isna().sum()
st.dataframe(nulls)
# Create a string buffer to capture the content
buffer = StringIO()
# Write the content into the buffer...use to_string
df.isna().sum().to_string(buffer)
# Retrieve the content from the buffer
null_values = buffer.getvalue()
# Use Streamlit to display the info
st.markdown("#### Null Values as String")
st.text(null_values)

# Function for producing histogram plot
def plot_hist(df, column):
    fig, ax = plt.subplots()
    sns.histplot(data=df,x=column)
    return fig
# Defining figure object
fig = plot_hist(df, "SalePrice")
st.markdown("#### Display of plt histogram")
st.pyplot(fig)

# Plot function to return countplot and info on categorical feature
def explore_categorical(df, x, fillna = True, placeholder = 'MISSING',
                        figsize = (6,4), order = None):
 
  # Make a copy of the dataframe and fillna 
  temp_df = df.copy()
  # Before filling nulls, save null value counts and percent for printing 
  null_count = temp_df[x].isna().sum()
  null_perc = null_count/len(temp_df)* 100
  # fillna with placeholder
  if fillna == True:
    temp_df[x] = temp_df[x].fillna(placeholder)
  # Create figure with desired figsize
  fig, ax = plt.subplots(figsize=figsize)
  # Plotting a count plot 
  sns.countplot(data=temp_df, x=x, ax=ax, order=order)
  # Rotate Tick Labels for long names
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
  # Add a title with the feature name included
  ax.set_title(f"Column: {x}")
  
  # Fix layout and show plot (before print statements)
  fig.tight_layout()
  plt.show()
    
  return fig, ax
# Define returned variables
fig, ax = explore_categorical(df, 'Alley')
st.markdown("#### Displaying a plot from explore_categorical function")
st.pyplot(fig)

def explore_numeric(df, x, figsize=(6,5) ):
  """Source: https://login.codingdojo.com/m/606/13765/117605"""
  # Making our figure with gridspec for subplots
  gridspec = {'height_ratios':[0.7,0.3]}
  fig, axes = plt.subplots(nrows=2, figsize=figsize,
                           sharex=True, gridspec_kw=gridspec)
  # Histogram on Top
  sns.histplot(data=df, x=x, ax=axes[0])
  # Boxplot on Bottom
  sns.boxplot(data=df, x=x, ax=axes[1])
  ## Adding a title
  axes[0].set_title(f"Column: {x}", fontweight='bold')
  ## Adjusting subplots to best fill Figure
  fig.tight_layout()
  # Ensure plot is shown before message
  plt.show()
  return fig
st.markdown("#### Displaying a plot from explore_numeric function")
fig = explore_numeric(df, 'Lot Frontage')
st.pyplot(fig)

# Final version of function
def plot_categorical_vs_target(df, x, y='SalePrice',figsize=(6,4),
                            fillna = True, placeholder = 'MISSING',
                            order = None):
  # Make a copy of the dataframe and fillna 
  temp_df = df.copy()
  # fillna with placeholder
  if fillna == True:
    temp_df[x] = temp_df[x].fillna(placeholder)
  
  # or drop nulls prevent unwanted 'nan' group in stripplot
  else:
    temp_df = temp_df.dropna(subset=[x]) 
  # Create the figure and subplots
  fig, ax = plt.subplots(figsize=figsize)
  
    # Barplot 
  sns.barplot(data=temp_df, x=x, y=y, ax=ax, order=order, alpha=0.6,
              linewidth=1, edgecolor='black', errorbar=None)
  
  # Boxplot
  sns.stripplot(data=temp_df, x=x, y=y, hue=x, ax=ax, 
                order=order, hue_order=order, legend=False,
                edgecolor='white', linewidth=0.5,
                size=3,zorder=0)
  # Rotate xlabels
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
  
  # Add a title
  ax.set_title(f"{x} vs. {y}")
  fig.tight_layout()
  return fig, ax

def plot_numeric_vs_target(df, x, y='SalePrice', figsize=(6,4), **kwargs): # kwargs for sns.regplot
  # Calculate the correlation
  corr = df[[x,y]].corr().round(2)
  r = corr.loc[x,y]
  # Plot the data
  fig, ax = plt.subplots(figsize=figsize)
  scatter_kws={'ec':'white','lw':1,'alpha':0.8}
  sns.regplot(data=df, x=x, y=y, ax=ax, scatter_kws=scatter_kws, **kwargs) # Included the new argument within the sns.regplot function
  ## Add the title with the correlation
  ax.set_title(f"{x} vs. {y} (r = {r})")
  # Make sure the plot is shown before the print statement
  plt.show()
  return fig, ax

# Add a selectbox for all possible features
column2 = st.selectbox(label="Select a column", options=columns_to_use, key="col2")
# Conditional statement to determine which function to use
if df[column2].dtype == 'object':
    fig, ax  = plot_categorical_vs_target(df, x=column2)
else:
    fig, ax = plot_numeric_vs_target(df, x=column2)
st.markdown("#### Explore Features vs Sales Price")
# Display appropriate eda plots
st.pyplot(fig)

import plotly.express as px
import plotly.io as pio
pio.templates.default='seaborn'

# Use plotly for explore functions
def plotly_explore_numeric(df, x):
    fig = px.histogram(df,x=x,marginal='box',title=f'Distribution of {x}', 
                      width=1000, height=500)
    return fig
def plotly_explore_categorical(df, x):
    fig = px.histogram(df,x=x,color=x,title=f'Distribution of {x}', 
                          width=1000, height=500)
    fig.update_layout(showlegend=False)
    return fig

# Add a selectbox for all possible features
column3 = st.selectbox(label="Select a column", options=columns_to_use, key="col3")
# Conditional statement to determine which function to use
if df[column3].dtype == 'object':
    fig = plotly_explore_categorical(df, column3)
else:
    fig = plotly_explore_numeric(df, column3)
    
st.markdown("#### Displaying appropriate Plotly plot based on selected column")
# Display appropriate eda plots
st.plotly_chart(fig)

# functionizing categoric vs target
def plotly_categorical_vs_target(df, x, y='SalePrice', histfunc='avg', width=800,height=500):
    fig = px.histogram(df, x=x,y=y, color=x, width=width, height=height,
                       histfunc=histfunc, title=f'Compare {histfunc.title()} {y} by {x}')
    fig.update_layout(showlegend=False)
    return fig

# functionizing numeric vs target
def plotly_numeric_vs_target(df, x, y = 'SalePrice', trendline = 'ols', add_hoverdata = True):
    if add_hoverdata == True:
        hover_data = list(df.columns)
    else: 
        hover_data = None
        
    pfig = px.scatter(df, x = x, y = y, width=800, height=600,
                     hover_data = hover_data,
                      trendline = trendline,
                      trendline_color_override = 'red',
                     title = f"{x} vs. {y}")
    
    pfig.update_traces(marker = dict(size=3),
                      line = dict(dash = 'dash'))
    pfig.update_layout(showlegend=False)
    return pfig

# Add a selectbox for all possible features
column4 = st.selectbox(label="Select a column", options=columns_to_use, key="col4")
# Conditional statement to determine which function to use
if df[column4].dtype == 'object':
    fig = plotly_categoric_vs_target(df, column4)
else:
    fig = plotly_numeric_vs_target(df, column4)
st.markdown("#### Plotly numeric or categoric")
# Display appropriate eda plots
st.plotly_chart(fig)





















