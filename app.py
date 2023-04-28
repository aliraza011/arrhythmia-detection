from flask import Flask, render_template,request
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np 

def classified_class(target_class):
    if target_class==0:
        return'Non-ecotic beats (normal beat) '
    if target_class==1:
        return'Supraventricular ectopic beats '
    if target_class==2:
        return'Ventricular ectopic beats '
    if target_class==3:
        return 'Fusion Beats '
    if target_class==4:
        return'Unknown Beats '
def Grad_cam(model, input_test,sample_number):
    
    array = np.array(input_test[sample_number])
    array1 = np.array(input_test[sample_number])
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    #print("array:",array.shape)   
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    #print("array:",array.shape) 
    predict = model.predict(array)
    per = max(predict[0])
    target_class = np.argmax(predict[0])
    name=classified_class(target_class)
    #print("Target Class = ", target_class, "corresponding to:", predict, "Obese is [0., 1.]")
    last_conv = model.get_layer('conv1d_4') #last_conv= model.layers[8]
    grad_model = tf.keras.models.Model([model.inputs], [last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(array) #get activations maps + predictions from last conv layer
        loss = predictions[:, target_class] # the variable loss gets the probability of belonging to the defined class (the predicted class on the model output)

    output = conv_outputs[0] #activations maps from last conv layer
    grads = tape.gradient(loss, conv_outputs) #function to obtain gradients from last conv layer

    #print("grads shape:", grads.shape)
    #print("Model output (loss for the target class):", loss.shape)
    #print("Output from lat conv layer", conv_outputs.shape)
    #print("Output from lat conv layer", conv_outputs.shape)
    pooled_grad= tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs=conv_outputs.numpy()
    pooled_grad = pooled_grad.numpy()
    for i in range(pooled_grad.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grad[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

   
    #Upsample the small_heatmap into a big_heatmap with cv2:
    big_heatmap = cv2.resize(heatmap, dsize=(188, 100), 
                         interpolation=cv2.INTER_CUBIC)
    reconstructed_ecg =  array/np.max(array)
    plt.imshow(big_heatmap, cmap='seismic', interpolation='lanczos')
    plt.colorbar()
    plt.plot((reconstructed_ecg[0]*50)+20,color='white',linewidth=2)


    plt.xlim(0,188)
    plt.ylim(0,100)
    plt.title(name)

    plt.savefig("static/draw.png")
    plt.clf()
    
    return big_heatmap,per


df=pd.read_csv('mitbih_test.csv')
X_test=df.iloc[:,:-1].values
X_test = X_test.reshape(len(X_test), X_test.shape[1],1)
X_test=(X_test-X_test.mean())/X_test.std()


app=Flask(__name__)

@app.route("/", methods=['POST','GET'])
def hello():
    re=request.method
    if re=='GET':
        return render_template('index.html')
    else:
        val=request.form['sample']
        val=int(val)
        x=X_test[val]
        plt.plot(x)
        plt.title('Origional ECG')
        plt.savefig('static/org.png')
        plt.clf()
        new_model = tf.keras.models.load_model('ecgclassifier')
        _,per=Grad_cam(new_model, X_test,val)
        a="Confidance Score:"+str(round((per*100),2))+"%"
        return render_template('index.html', href='static/draw.png', href2='static/org.png',con=a)
if __name__=='__main__':
    from werkzeug.serving import run_simple
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    run_simple('localhost', 5000, app)
  
          
           
