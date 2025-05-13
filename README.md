# EXPERIMENT NO: 07
# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries: pandas, matplotlib.pyplot, and KMeans from sklearn.

2. Load the dataset using pandas read_csv().

3. Display the first few records of the dataset using head().

4. Print dataset structure using info() and check for missing values with isnull().sum().

5. Initialize an empty list to store Within Cluster Sum of Squares (WCSS) values.

6. Loop through cluster numbers from 1 to 10:
   
     a. Initialize the KMeans model with the current number of clusters.
   
     b. Fit the model on the selected features (Annual Income and Spending Score).
   
     c. Append the model's inertia (WCSS) to the list.

8. Plot the WCSS values to find the "elbow" point that suggests the optimal number of clusters.

9. Initialize and fit the KMeans model using the chosen number of clusters (e.g., 5).

10. Predict the clusters and store the cluster labels in a new column in the dataset.

11. Create separate DataFrames for each cluster based on predicted labels.

12. Plot each cluster on a scatter plot using different colors and add a legend.

13. Set the title and axis labels for the scatter plot to visualize customer segments.
 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: HARI PRIYA M
RegisterNumber: 212224240047  
*/
```

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    customer_data = pd.read_csv("/content/Mall_Customers.csv")
    print(customer_data.head())
    
    print(customer_data.info())

    print(customer_data.isnull().sum())

    within_cluster_sums = []
    for k in range(1, 11):
        model = KMeans(n_clusters=k, init="k-means++", random_state=42)
        model.fit(customer_data.iloc[:, 3:])
        within_cluster_sums.append(model.inertia_) 
    
    plt.plot(range(1, 11), within_cluster_sums)
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method for Optimal Clusters")
    plt.show()
    
    kmeans_model = KMeans(n_clusters=5, random_state=42)
    kmeans_model.fit(customer_data.iloc[:, 3:])
    
    predicted_clusters = kmeans_model.predict(customer_data.iloc[:, 3:])
    predicted_clusters
    
    customer_data["Segment"] = predicted_clusters
    segment_0 = customer_data[customer_data["Segment"] == 0]
    segment_1 = customer_data[customer_data["Segment"] == 1]
    segment_2 = customer_data[customer_data["Segment"] == 2]
    segment_3 = customer_data[customer_data["Segment"] == 3]
    segment_4 = customer_data[customer_data["Segment"] == 4]
    
    plt.scatter(segment_0["Annual Income (k$)"], segment_0["Spending Score (1-100)"], c="red", label="Segment 0")
    plt.scatter(segment_1["Annual Income (k$)"], segment_1["Spending Score (1-100)"], c="green", label="Segment 1")
    plt.scatter(segment_2["Annual Income (k$)"], segment_2["Spending Score (1-100)"], c="blue", label="Segment 2")
    plt.scatter(segment_3["Annual Income (k$)"], segment_3["Spending Score (1-100)"], c="black", label="Segment 3")
    plt.scatter(segment_4["Annual Income (k$)"], segment_4["Spending Score (1-100)"], c="yellow", label="Segment 4")
    plt.legend()
    plt.title("Customer Segments Visualization")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.show()

## Output:

Displaying

customer_data.head()

![Screenshot 2025-05-13 083908](https://github.com/user-attachments/assets/bcb6ce08-384b-4c19-83c8-77c410a9e8f0)

customer_data.info()

![Screenshot 2025-05-13 083901](https://github.com/user-attachments/assets/ea0ccfde-f35c-4175-bb64-442d57df5a26)

customer_data.isnull().sum()

![Screenshot 2025-05-13 083853](https://github.com/user-attachments/assets/8164efd9-b390-4019-a882-6e8ba66fe614)

Plotting using Elbow Method

![Screenshot 2025-05-13 083846](https://github.com/user-attachments/assets/36554249-859c-4e8d-91d7-a9181201760e)

K=means Clustering

![Screenshot 2025-05-13 083839](https://github.com/user-attachments/assets/dbf0af46-d405-4ff4-9c4c-f4ef4453aaeb)

Predicted Clusters

![Screenshot 2025-05-13 083831](https://github.com/user-attachments/assets/bc2fb1ee-815f-4b03-a9cb-2c9ff74b2038)

Plotting the clusters : Cusomer Segment

![Screenshot 2025-05-13 085040](https://github.com/user-attachments/assets/8e8bdbf1-f1a6-4e15-ac43-1ab8411770ca)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
