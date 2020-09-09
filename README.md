# **Traffic Signs Classification**

**Objective and Problem Statement**
- Traffic Signs Classification is an important task for self driving cars. In this Project, I have prepared a Deep Convolutional Neural Network Model which can classify the images of 43 distinct types of Traffic Signals. It is a multi-class Classification Project. You can gain insights about the Impelementation of Deep Convolutional Neural Network in Image Classification.

**Convolutional Neural Network**
- In deep learning, a convolutional neural network is a class of deep neural networks, most commonly applied to analyzing visual imagery. They are also known as shift invariant or space invariant artificial neural networks, based on their shared-weights architecture and translation invariance characteristics.

**Getting the Data**
- I have manually downloaded the Data from [Kaggle](https://www.kaggle.com/). You can download the [Data](https://github.com/ThinamXx/TrafficSigns..Classification/blob/master/TrafficSign%20Data.rar) from [here](https://github.com/ThinamXx/TrafficSigns..Classification/blob/master/TrafficSign%20Data.rar). I have used Google Colab for this Project so the act of reading and loading the Data might be different in various other platforms.
- The Dataset contains 43 different types of Traffic Signs. The overview of the each Signs are summarized below:

**Index** | **Traffic Signs**
--------- | -----------------
0 | Speed Limit (20km/h)
1 | Speed Limit (30km/h)
2 | Speed Limit (50km/h)
3 | Speed Limit (60km/h)
4 | Speed Limit (70km/h)
5 | Speed Limit (80km/h)
6 | End of Speed Limit (80km/h)
7 | Speed Limit (100km/h)
8 | Speed Limit (120km/h)
9 | No Passing
10 | No Passing for the Vehicles over 3.5 metric tons
11 | Right of way at next intersection
12 | Priority Road
13 | Yield
14 | Stop
15 | No Vehicles
16 | Vehicles over 3.5 metric tons prohibited
17 | No Entry
18 | General Caution 
19 | Dangerous curve to the left
20 | Dangerous curve to the right
21 | Double Curve
22 | Bumpy Road
23 | Slippery Road
24 | Road narrows on the right
25 | Road Work
26 | Traffic Signals
27 | Pedestrains
28 | Children Crossing
29 | Bicycles Crossing
30 | Beware of Ice/Snow
31 | Wild animals crossing
32 | End of all speed and Passing limits
33 | Turn right ahead
34 | Turn left ahead
35 | Ahead Only
36 | Go straight or right
37 | Go straight or left
38 | Keep Right
39 | Keep Left
40 | Round about Mandatory
41 | End of no Passing
42 | End of no Passing by vehicles over 3.5 metric tons.

**Snapshot of the Input Images**

![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1599625942/Traffic_ztr4fa.png)

**Convolutional Neural Network**
- In deep learning, a convolutional neural network is a class of deep neural networks, most commonly applied to analyzing visual imagery. They are also known as shift invariant or space invariant artificial neural networks, based on their shared-weights architecture and translation invariance characteristics.

```javascript
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(32, 32, 1)),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation="relu"),
                                    tf.keras.layers.Dense(120, activation="relu"),
                                    tf.keras.layers.Dense(84, activation="relu"),
                                    tf.keras.layers.Dense(43, activation="softmax")                                   
])
```

**Model Evaluation**

  - Plotting Training Accuracy vs Validation Accuracy
  
  ![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1599626424/Hey_heil1n.png)
  
  - Plotting Training Loss vs Validation Loss
  
  ![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1599626504/Hey2_etpkcf.png)

**Snapshot of the Predicted Image**

![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1599626599/Predict_nn167z.png)
