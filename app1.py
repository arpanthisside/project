from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import io
import base64

app = Flask(__name__)
STATIC_FOLDER = 'D:\\personal codes\\project\\static'
os.makedirs(STATIC_FOLDER, exist_ok=True)

data = pd.read_csv(r"D:\personal codes\project\uploads\cluster farmland 6MAR (1).csv")

def process_data(selected_features):
    global data
    scaler = MinMaxScaler()
    data['Land_Area_Norm'] = scaler.fit_transform(data[['Land_Area']])
    data['Water_Availability_Norm'] = scaler.fit_transform(data[['Water_Availability']])
    data['Market_Access_Norm'] = scaler.fit_transform(data[['Market_Access']])
    data['Annual_Income_Norm'] = scaler.fit_transform(data[['Annual_Income']])
    
    if 'Soil_Compatibility' not in data.columns:
        data['Soil_Compatibility'] = np.random.rand(len(data))
    
    principal_features = data[['Land_Area_Norm', 'Soil_Compatibility', 'Water_Availability_Norm', 'Market_Access_Norm', 'Annual_Income_Norm']]
    non_principal_features = data[['Location', 'Crop_Type', 'Farming_Method', 'Fertilizer_Used', 'Machinery_Availability']]
    
    encoder = OneHotEncoder(sparse_output=False)
    non_principal_encoded = encoder.fit_transform(non_principal_features)
    scaler = StandardScaler()
    non_principal_scaled = scaler.fit_transform(non_principal_encoded)
    
    pca = PCA(n_components=0.95)
    non_principal_pca = pca.fit_transform(non_principal_scaled)
    non_principal_pca_df = pd.DataFrame(non_principal_pca, columns=[f'PC{i+1}' for i in range(non_principal_pca.shape[1])])
    
    combined_features = pd.concat([principal_features, non_principal_pca_df], axis=1)
    
    if selected_features:
        combined_features = combined_features[selected_features]
    
    def coalition_strength_score(row):
        w1, w2, w3, w4, w5 = 0.3, 0.2, 0.2, 0.15, 0.15
        return (
            w1 * row["Land_Area_Norm"] +
            w2 * row["Soil_Compatibility"] +
            w3 * row["Water_Availability_Norm"] +
            w4 * row["Market_Access_Norm"] +
            w5 * row["Annual_Income_Norm"]
        )
    
    data['Coalition_Score'] = data.apply(coalition_strength_score, axis=1)
    
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    coalitions = kmeans.fit_predict(combined_features)
    data['Coalition'] = coalitions
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Generate scatter plot
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(combined_features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data['Coalition'], cmap='viridis', s=50, alpha=0.6)
    plt.title('2D Scatter Plot of Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Cluster')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    scatter_plot = base64.b64encode(img.getvalue()).decode('utf-8')

    # Generate radar chart
    cluster_profiles = data.groupby('Coalition')[selected_features].mean()
    labels = selected_features
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for cluster, profile in cluster_profiles.iterrows():
        values = profile.tolist()
        values += values[:1]
        ax.plot(angles, values, label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.title('Radar Chart of Cluster Profiles')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    radar_chart = base64.b64encode(img.getvalue()).decode('utf-8')

    return scatter_plot, radar_chart, data[['Farmer_ID', 'Coalition', 'Coalition_Score']]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('process1'))
    return render_template('index.html')

@app.route('/process1', methods=['GET', 'POST'])
def process1():
    principal_features = ['Land_Area_Norm', 'Soil_Compatibility', 'Water_Availability_Norm', 'Market_Access_Norm', 'Annual_Income_Norm']
    if request.method == 'POST':
        selected_features = request.form.getlist('features')
        scatter_plot, radar_chart, clustered_data = process_data(selected_features)
        return render_template('process1.html', scatter_plot=scatter_plot, radar_chart=radar_chart, clustered_data=clustered_data.to_html(), selected_features=selected_features, principal_features=principal_features)
    return render_template('process1.html', principal_features=principal_features)

if __name__ == '__main__':
    app.run(debug=True)