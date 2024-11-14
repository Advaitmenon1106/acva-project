import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from predict_segmentation import predict_tumour, visualize_prediction
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import streamlit as st

# Load models
image_classification_model = YOLO('./best_model.pt')
segmentation_model = load_model('unet_model.h5')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# # Load text generation models and data
# df, text_model, index, generator_model, gen_tokenizer = load_text_generation_models()

# Streamlit app layout
st.title('Multi-task Prediction App')

#"Text Generation"
task = st.selectbox(
    "Select a Task",
    ("Image Classification", "Image Segmentation", "Experimental Segmentation")
)

if task == "Image Classification":
    st.header('Image Classification')
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        input_img = np.array(Image.open(uploaded_file))
        pred = image_classification_model.predict(input_img)
        for box in pred[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            confidence = box.conf[0]                # Get confidence score
            
            # Draw bounding box and label on image
            cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(input_img, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        st.image(input_img)

elif task == "Image Segmentation":
    st.header("Image Segmentation")
    uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        pred_mask = predict_tumour(uploaded_file, segmentation_model)
        visualize_prediction(uploaded_file, pred_mask)
        st.image('predicted_mask_output.png')

elif task == 'Experimental Segmentation':

# Define the TensorFlow model equivalent to MyNet in PyTorch
    class MyNetTF(Model):
        def __init__(self, input_dim, nChannel, nConv):
            super(MyNetTF, self).__init__()
            self.conv1 = layers.Conv2D(nChannel, kernel_size=3, strides=1, padding='same', input_shape=(None, None, input_dim))
            self.bn1 = layers.BatchNormalization()

            # Create lists to store intermediate Conv and BatchNorm layers
            self.conv2 = [layers.Conv2D(nChannel, kernel_size=3, strides=1, padding='same') for _ in range(nConv - 1)]
            self.bn2 = [layers.BatchNormalization() for _ in range(nConv - 1)]

            self.conv3 = layers.Conv2D(nChannel, kernel_size=1, strides=1, padding='same')
            self.bn3 = layers.BatchNormalization()

        def call(self, x):
            x = self.conv1(x)
            x = tf.nn.relu(x)
            x = self.bn1(x)
            
            for i in range(len(self.conv2)):
                x = self.conv2[i](x)
                x = tf.nn.relu(x)
                x = self.bn2[i](x)
            
            x = self.conv3(x)
            x = self.bn3(x)
            return x

    # Streamlit inputs for model parameters
    img_upload = st.file_uploader('Upload an image')
    visualise=False

    if img_upload is not None:
        # Read the image file as bytes
        file_bytes = np.asarray(bytearray(img_upload.read()), dtype=np.uint8)
        im = cv2.imdecode(file_bytes, 1)

        nChannel = st.number_input('Choose the number of channels', step=1, value=100)
        nConv = st.number_input('Choose the number of convolution layers in the network', step=1, value=3)
        maxIter = st.number_input('Choose the number of epochs', step=1, value=40)
        lr = st.number_input('Choose the learning rate', value=0.1)
        min_labels = st.number_input('Choose the number of segments you wish to see', value=3)

        start_button = st.button('Click to start the segmentation')
        if start_button:
            # Process the image and normalize it
            im = im.astype('float32') / 255.0
            data = np.expand_dims(im, axis=0)  # Add batch dimension

            # Initialize and compile the TensorFlow model
            model = MyNetTF(data.shape[-1], nChannel, nConv)
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
            label_colours = np.random.randint(255, size=(100, 3))

            # Define custom losses for segmentation
            def continuity_loss(output):
                # Compute horizontal and vertical differences in the output feature map
                HPy = output[:, 1:, :, :] - output[:, :-1, :, :]
                HPz = output[:, :, 1:, :] - output[:, :, :-1, :]
                return tf.reduce_mean(tf.abs(HPy)) + tf.reduce_mean(tf.abs(HPz))

            # Training loop
            for batch_idx in range(int(maxIter)):
                with tf.GradientTape() as tape:
                    output = model(data)  # Forward pass

                    # Cross-entropy loss
                    output_reshaped = tf.reshape(output, [-1, nChannel])
                    target = tf.argmax(output_reshaped, axis=1)
                    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_reshaped, labels=target)
                    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)

                    # Continuity loss
                    continuity = continuity_loss(output)

                    # Total loss
                    loss = cross_entropy_loss + continuity

                # Backpropagation and optimization
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                # Get the predicted label
                predicted_label = tf.argmax(tf.reshape(output, [-1, nChannel]), axis=1)
                im_target_rgb = np.array([label_colours[c % nChannel] for c in predicted_label.numpy()])
                im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)

                # Display the output if visualization is set
                
                st.image(im_target_rgb)
                
                    
                print(f"Batch {batch_idx + 1}/{maxIter}, Loss: {loss.numpy()}, Labels: {len(np.unique(predicted_label))}")

                if len(np.unique(predicted_label)) <= min_labels:
                    print("Target number of labels reached.")
                    break
