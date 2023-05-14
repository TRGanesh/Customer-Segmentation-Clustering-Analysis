# IMPORTING REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as py
import plotly.graph_objects as go
import time
import pickle
import streamlit as st
from streamlit_lottie import st_lottie
import streamlit_option_menu
import requests
import json
from PIL import Image
from streamlit_extras.dataframe_explorer import dataframe_explorer

# SETTING PAGE CONFIGURATION
st.set_page_config(page_title='Customer Segmentation',layout='wide')

# SETTING STREAMLIT STYLE
streamlit_style = """   <style>
                        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');
                        
                        html,body,[class*='css']{
                            font-family:'serif';
                        }
                        </style>
                  """
st.markdown(streamlit_style,unsafe_allow_html=True)


# LOADING LOTTIE FILES
def load_lottier(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# LOADING DATASET
df = pd.read_csv('marketing_campaign.csv.xls',sep='\t')
df.drop(columns=['Z_CostContact','Z_Revenue'],inplace=True)

def main():
    
    # USING LOCAL CSS
    def local_css(css_filename):
        with open(css_filename) as f:
            st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
    local_css('style.css')     
    
    # CREATING NAVIGATION BAR WITH OPTION MENU 
    selected_page = streamlit_option_menu.option_menu(menu_title=None,
                                                      options=['Data','Analysis'],
                                                      icons=['activity','bar-chart'],
                                                      menu_icon='list',default_index=0,
                                                      orientation='horizontal',
                                                      styles={
                                                          'container':        {'padding':'0!important',             'background-color':'#white'},
                                                          
                                                          'icon':{'color':'yellow','fontsize':'25px'},
                                                          
                                                          'nav-link':{'fontsize':'25px',
                                                                      'text-align':'middle',
                                                                      'margin':'0px',
                                                                      '--hover-color':'grey'},
                                                          
                                                          'nav-link-selected':{'background-color':'blue'}
                                                      })  
    
    def title(string):
        st.markdown(f"<h1 style='color:#e8322e';font-size:40px>{string}</h1>",unsafe_allow_html=True)
    def header(string):
        st.markdown(f"<h2 style='color:#FF00FF';font-size:40px>{string}</h2>",unsafe_allow_html=True)
    def subheader(string):
        st.markdown(f"<h3 style='color:#e5f55b';font-size:40px>{string}</h3>",unsafe_allow_html=True)
    def plot_subheader(string):
        st.markdown(f"<h3 style='color:#41FB3A';font-size:40px>{string}</h3>",unsafe_allow_html=True) 
    def inference_subheader(string):
        st.markdown(f"<h4 style='color:#9933ff';font-size:40px>{string}</h4>",unsafe_allow_html=True)
    def plot_subheader2(string):
        st.markdown(f"<h4 style='color:#ffff80';font-size:40px>{string}</h4>",unsafe_allow_html=True)                               

    # CREATING DATA PAGE
    if selected_page == 'Data':
        
        title('Unsupervised Machine Learning')
        with st.container():
            text_column,image_column = st.columns((2,1))
            with text_column:
                st.write("Machine Learning can be divided majorly into **:orange[Supervised Machine Learning]**,**:orange[Unsupervised Machine Learning]**,**:orange[Semi-Supervised Machine Learning]**,**:orange[Reinforcement Machine Learning]**.In Supervised Machine Learning,models get trained by using labeled data.That means,model gets trained by X features and Y target.But it's not the case with Unsupervised Machine Learning,here the data is not pre-labeled.Here,models do find hidden patterns and insights from data.The goal of Unsupervised Machine Learning is to find the underlying structure of dataset,group the data according to the similarities and to represent the data in a compressed format.")
                st.write("Some of the Unsupervised Machine Learning Algorithms are **:orange[K-Means Clustering]**,**:orange[Hierarchical Clustering]**,**:orange[Principal Component Analysis(Dimensionality Reduction Technique)]**.")
            with image_column:
                st.image(Image.open('Unsupervised_ML_image.png.webp'),width=650,use_column_width=True)
                
                    
        header('Customer Segmentation')
        with st.container():
            text_column,image_column = st.columns((2,1))
            with text_column:
                st.write(' ')
                subheader('What is Customer Segmentation')
                st.write(' ')
                st.write(' ')
                st.write("Customer Segmentation simply means grouping customers according to various characteristics(for example grouping customers by age).It's a way for Business Organizations to understand thier customers.Knowing the differences between customer groups,it's easier to make strategic decisions regarding the product growth and marketing.It can also be said as Customer Personality Analysis.")
            with image_column:
                st.image(Image.open('customer_segmentation_image.png.jpeg'),use_column_width=False,width=400)
                    
        # DISPLAYING DATASET
        header('Dataset')
        st.write("Below is a dataset having data related to customers of a Marketing Campaign of a Company.")
        st.write('You can download the dataset from Kaggle from [here](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)')
        st.dataframe(dataframe_explorer(df),use_container_width=True)
        st.write('- - -')
        
        # FEATURES OF DATASET
        header('Features of Dataset')
        description = ["Customer's unique identifier",
                       "Customer's birth year",
                       "Customer's education level",
                       "Customer's marital status",
                       "Customer's yearly household income",
                       "Number of children in customer's household",
                       "Number of teenagers in customer's household",
                       "Date of customer's enrollment with the company",
                       "Number of days since customer's last purchase",
                       "Amount spent on wine in last 2 years",
                       "Amount spent on fruits in last 2 years",
                       "Amount spent on meat in last 2 years",
                       "Amount spent on fish in last 2 years",
                       "Amount spent on sweets in last 2 years",
                       "Amount spent on gold in last 2 years",
                       "Number of purchases made with a discount",
                       "Number of purchases made through the company’s website",
                       "Number of purchases made using a catalogue",
                       "Number of purchases made directly in stores",
                       "Number of visits to company’s website in the last month",
                       "1 if customer accepted the offer in the 3rd campaign, 0 otherwise",
                       "1 if customer accepted the offer in the 4rth campaign, 0 otherwise",
                       "1 if customer accepted the offer in the 5th campaign, 0 otherwise",
                       "1 if customer accepted the offer in the 1st campaign, 0 otherwise",
                       "1 if customer accepted the offer in the 2nd campaign, 0 otherwise",
                       "1 if the customer complained in the last 2 years, 0 otherwise",
                       "1 if customer accepted the offer in the last campaign, 0 otherwise"]

        feature_description_df = pd.DataFrame({'Feature Name':df.columns.tolist(),
                                               'Description':description},index=range(1,28))
        st.table(feature_description_df)
        st.write('- - -')
        
        header('3D Visualization of Clusters')
        # READING DATAFRAME
        principal_components_df = pd.read_csv('principal_components_customer_seg.csv')  
        principal_components_df['Clusters'] = principal_components_df['Clusters'].astype('O')
        # PLOTTING 3D SCATTER PLOT 
        with st.container():
            
            fig = px.scatter_3d(data_frame=principal_components_df,x='Principal Component 1',y='Principal Component 2',z='Principal Component 3',color='Clusters',width=500,height=700,template='plotly_dark')
            # UPDATING THE FIGURE
            fig.update_layout({'paper_bgcolor':'rgb(255,255,230)'},
                              font_color='black',
                              hoverlabel=dict(bgcolor='black',
                                              font_family='Rockwell',
                                              font_color='white'),
                              legend_title='Cluster',
                              legend=dict(bgcolor='rgb(255,255,230)',
                                          yanchor='top',y=0.79,
                                          xanchor='left',x=0.97,
                                          font=dict(size=20,color='black')))
            fig.update_layout(legend_title_font_size=20,legend_title_font_color='black')
            #fig.update(layout_coloraxis_showscale=False)
            
            st.plotly_chart(fig,use_container_width=True)      
        st.write('- - -')
        subheader('Steps Followed:')
        st.write('- I Imported required Python libraries')
        st.write('- I did Data Exploration,Visualization using Matplotlib,Seaborn,Plotly')
        st.write('- I did Feature Extraction from given features')
        st.write('- And then dealing with Outliers is done')
        st.write('- I did Categorical Feature Encoding,beacause Machine Learning model takes only numeric features')
        st.write("- In K-Means Clustering,I used Elbow Method and YellowBrick's KElbow Visualizer for getting correct number of clusters(3)")
        st.write('-  Then I used Agglomerative Clustering - Hierarchical Clustering for Cluster prediction of data')
        st.write('- Then,I plotted features based on Clusters')
        st.write('- For Cluster Visualization,i used Dimensionality Reduction Technique - Principal Component Analysis with 3 Principal Components')
        st.write('- - -')
        st.write('In Analysis page,You can get an overview about K_Means Clustering,Hierarchical Clustering,Principal Component Analysis.Also,You can see some plots.')
        
    if selected_page == 'Analysis':
        header('K-Means Clustering')
        with st.container():
            text_column,image_column = st.columns((2,1))
            with text_column:
                st.write(' ')
                st.write("K-Means Clustering algorithm groups the unlabeled data into different clusters.K denotes the number of clusters.It is a **Centroid-based** algorithm,where each cluster has a centriod.Main aim is to reduce the sum of squared distances between data points and their corresponding clusters i.e **:orange[With-in Cluster Sum of Squares(WCSS)]** distance.Consequently,maximizes the distance between cluster-cluster.")
                st.write("For getting optimal number of clusters,we use Elbow Method.In that method,for each value of K,it calculates WCSS distance.We can plot WCSS with K value,the plot looks like an Elbow.")
            with image_column:
                k_means_image = Image.open('k-means_image.png')
                st.image(k_means_image)
        
        header('Hierarchical Clustering')
        with st.container():
            text_column,image_column = st.columns((2,1))
            with text_column:
                st.write(' ')
                st.write("Hierarchical Clustering is also known as Hierarchical Cluster Analysis.Here,we develop hierarchy of clusters in form of a tree.This tree-shaped structure is called **:orange[Dendrogram]**.")
                st.write("I used Agglomarative Clustering approach for hierarchical clustering,it is a bottom-up approach,in which algorithm starts with taking all data points as single clusters and then starts mergeing closest pair of clusters together.It does this until all clusters are merged into a single cluster having all data points.")
            with image_column:
                st.image(Image.open('hierarchical_clustering_image.jpeg'),width=400,use_column_width=True)    
        
        header('Principal Component Analysis') 
        with st.container():
               text_column,image_column = st.columns((2,1))
               with text_column:
                    st.write(' ')
                    st.write("Principal Component Analysis is an unsupervised learning algorithm that is used for dimensionality reduction.It is a statistical process that converts observations of correlated features into a set of linearly uncorrelated features with the help of Orthogonal Transformation.Newly transformed features are called Principal Components.")
                    st.write("It tries to project higher dimensional data into lower dimensional data.Dimensionality Reduction is a type of feature extraction technique that aims to reduce the number of input features while retaining as much of the original information as possible.")
               with image_column:
                    st.image(Image.open('principal_comp_analysis_image.png.jpeg'),use_column_width=True)
        st.write('- - -')
        
        st.write("Let's see some plots based on clustering")
        # LOADING THE DATASET WHICH HAS NON-TRANSFORMED FEATURES WITH CLUSTER LABELS
        
        df_with_pc_clusters = pd.read_csv('df_PC_customer_seg.csv')
        
        df_with_pc_clusters['Cluster'] = df_with_pc_clusters['Cluster'].astype('O')
        # SCATTERPLOT OF INCOME AND AMOUNT SPENT BASED ON CLUSTERS
        
        with st.container():
                plot_subheader2('Clusters based on the Scatter Plot between Income and Amount Spent by Customers.')
                fig = px.scatter(data_frame=df_with_pc_clusters,x='Amount Spent',y='Income',color='Cluster',width=1700,height=700,color_discrete_sequence=px.colors.qualitative.Light24)
                 
                fig.update_layout(hoverlabel=dict(bgcolor='black',
                                              font_size=16,
                                              font_family='Rockwell',
                                              font_color='white'))
                #fig.update(layout_coloraxis_showscale=False)
                fig.update_layout(legend_title='Cluster',
                                  legend=dict(font=dict(size=20)))
                fig.update_traces(textfont_size=20)
                fig.update_layout(legend_title_font_size=20)
                
                st.plotly_chart(fig,use_container_width=True) 
                
                inference_subheader('Inference') 
                st.write('From above Scatter plot,we can observe,')
                st.write('- Customers who are belonging to 0th Cluster,their Amount Spent ranges from 8 to 230,and Income ranges from 7k to 50k')
                st.write('- Customers who are belonging to 2nd Cluster,thier Amount Spent ranges from 320 to 930 and Income ranges from 45k to 70k')
                st.write('- Customers who are belonging to 1st Cluster,thier Amount Spent ranges roughly from 1200 to 1920 and Income ranges from 70k to 90k')
                st.write('- Also,we can see more overlapping between Cluster2 and Cluster1 data points')
        st.write('- - -')        
        with st.container():
            plot_subheader2(('Distribution plots of Amount Spent and Income of Customer based on their Cluster'))
            fig,axes = plt.subplots(1,2,dpi=280,figsize=(13,4))
            sns.set(rc={"axes.facecolor":"#000000","figure.facecolor":"#ccffff"})
            sns.boxenplot(data=df_with_pc_clusters,y='Amount Spent',x='Cluster',palette='Spectral_r',ax=axes[0])
            axes[0].grid(False)
            sns.boxenplot(data=df_with_pc_clusters,y='Income',x='Cluster',palette='Spectral_r',ax=axes[1])
            axes[1].grid(False)
            plt.tight_layout()
            st.pyplot(fig)       
            
            inference_subheader('Inference') 
            st.write('From above Boxen plots,we can observe')
            st.write("- In each cluster group,the distribution's Inter-Quartile range(the rectangle in middle) is not overlapping.")
            st.write('- Inter-Quartile range tells us the spread of middle half of the distribution.Basically,it is the difference between First quartile and Third quartile.')
                 
if __name__ == '__main__':
    main()
