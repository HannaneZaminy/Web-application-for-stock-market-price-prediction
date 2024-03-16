import streamlit as st
import pandas as pd
import plotly.express as px
st.set_page_config(page_title="Analiz" , page_icon=":bar_chart:" ,layout="wide")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
st.write("""
# Analiz Your Dream House
""")
st.write('___')
df= pd.read_csv("HousePrice.csv")
col1, col2 = st.columns((2))
df.drop_duplicates()
df=df.dropna().reset_index(drop=True)
df=df[['Area','Room','Parking','Warehouse','Address','PriceMC']]
percentile25 = df['PriceMC'].quantile(0.25)
percentile75 = df['PriceMC'].quantile(0.75)
iqr=percentile75 -percentile25
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
new_df =df[df['PriceMC'] < upper_limit]
new_df=new_df[new_df['PriceMC'] > lower_limit]
st.dataframe(new_df)
st.sidebar.header("Choose your filter: ")

region=st.sidebar.multiselect("Pick your Region" ,df["Address"].unique())
if not region:
    df2=df.copy()
else:
    df2=df[df["Address"].isin(region)]

room=st.sidebar.multiselect("Pick the Room " , df2["Room"].unique())
if not room:
    df3 =df2.copy()
else:
    df3=df2[df2["Room"].isin(room)]

if not region and not room:
    filtered_df =df
elif not room:
    filtered_df =df[df["Address"].isin(region)]
elif not region:
    filtered_df=df[df["Room"].isin(room)]
else:
# elif room and region:
    filtered_df =df3[df["Room"].isin() & df3["Address"].isin()]

category_df = filtered_df.groupby(by = ["Address"], as_index = False)["PriceMC"].sum()
with col1:
    st.subheader("Price Category by Region")
    fig = px.bar(category_df, x = "Address", y = "PriceMC", text = ['${:,.2f}'.format(x) for x in category_df["PriceMC"]],
             )
    st.plotly_chart(fig,use_container_width=True, height = 1500)

with col2:
    st.subheader("Houses in the Region ")
    fig = px.pie(filtered_df, values = "Address", names = "Address", hole = 0.25)
    fig.update_traces(text = filtered_df["PriceMC"], textposition = "outside")
    st.plotly_chart(fig,use_container_width=True)