# IdenProf
IdenProf is a dataset containing images of identifiable professionals.
<hr>
<b>IdenProf</b> is a dataset of identifiable professionals, collected in order to ensure that machine learning systems can be trained
 to recognize professionals by their mode of dressing as humans can observe. This is part of our mission to train machine learning systems to 
  perceive, understand and act accordingly in any environment they are deployed. <br><br>

  This is the first release of the IdenProf dataset. It contains 11,000 images that span cover 10 categories of professions. The professions 
  included in this release are: <br><br>

  - <b> Chef </b> <br>
  - <b> Doctor </b> <br>
  - <b> Engineer </b> <br>
  - <b> Farmer </b> <br>
  - <b> Firefighter </b> <br>
  - <b> Judge </b> <br>
  - <b> Mechanic </b> <br>
  - <b> Pilot </b> <br>
  - <b> Police </b> <br>
  - <b> Waiter </b> <br> <br>

  There are <b>1,100 images</b> for each category, with <b>900 images for trainings </b> and <b>200 images for testing</b> . We are working on adding more
   categories in the future and will continue to improve the dataset.
  <br><br> <br> <br>

  <b>>>> DOWNLOAD, TRAINING AND PREDICTION: </b> <br><br>
 The <b>IdenProf</b> dataset is provided for download in the <b>release</b> section of this repository.
 You can download the dataset via this <a href="https://github.com/OlafenwaMoses/IdenProf/releases/" >link</a> . <br><br>

 We have also provided a python codebase to download the images, train <b>ResNet50</b> on the images
  and perform prediction using a pretrained model (also using <b>ResNet50</b>) provided in the release section of this repository.
  The python codebase is contained in the <b><a href="idenprof.py" >idenprof.py</a></b> file and the model class labels for prediction is also provided the 
  <b><a href="idenprof_model_class.json" >idenprof_model_class.json</a></b>. The pretrained <b>ResNet50</b> model is available for download via this 
  <b><a href="https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof_061-0.7933.h5" >link</a></b>. This pre-trained model was trained over **61 epochs** only, but it achieved **79%** accuracy on 2000 test images. You can see the prediction results on new images that were not part of the dataset in the **Prediction Results** section below. More experiments will enhance the accuracy of the model.
<br>
Running the experiment or prediction requires that you have **Tensorflow**, **Numpy** and **Keras** installed.
<br><br> <br> <br>

  <b>>>> DATASHEET FOR IDENPROF</b> <br><br>
  For transparency and accountability on the collection and content of the IdenProf dataset, we have provided a comprehensive
   datasheet on the dataset . The datasheet is based on the blueprint provided in the 2018 paper publication , "Datasheets for Datasets" by Timnit. et al.
    The datasheet is available via this <b><a href="idenprof-datasheet.pdf" >link</a></b>. <br><br>

<b>>>> Prediction Results</b> <br><br>
  <img src="test-images/1.jpg" />
<pre>
chef  :  99.90828037261963
waiter  :  0.0905417778994888
doctor  :  0.0011116820132883731
</pre>

<hr>
<br>
<img src="test-images/2.jpg" />
<pre>
firefighter  :  80.1691472530365
police  :  19.79282945394516
engineer  :  0.03719799569807947
</pre>

<hr>
<br>

<img src="test-images/3.jpg" />
<pre>
farmer  :  99.93320107460022
police  :  0.06526767974719405
firefighter  :  0.0014684919733554125
</pre>

<hr>
<br>

<img src="test-images/4.jpg" />
<pre>
doctor  :  99.70111846923828
chef  :  0.2974770264700055
waiter  :  0.001407588024449069
</pre>

<hr>
<br>

<img src="test-images/5.jpg" />
<pre>
waiter  :  99.99997615814209
chef  :  1.568847380895022e-05
judge  :  1.0255866556008186e-05
</pre>

<hr>
<br>

<img src="test-images/6.jpg" />
<pre>
pilot  :  99.75990653038025
mechanic  :  0.21259593777358532
police  :  0.024273521557915956
</pre>

<hr>
<br>

<img src="test-images/7.jpeg" />
<pre>
farmer  :  100.0
waiter  :  1.6071012576279742e-09
police  :  1.273151375991155e-09
</pre>

<hr>
<br>

<img src="test-images/8.jpg" />
<pre>
doctor  :  95.55137157440186
engineer  :  3.5533107817173004
mechanic  :  0.6231860723346472
</pre>

<hr>
<br>

<img src="test-images/9.jpg" />
<pre>
waiter  :  99.92395639419556
chef  :  0.05305781960487366
judge  :  0.01294929679716006
</pre>

<hr>
<br>

<img src="test-images/10.jpg" />
<pre>
police  :  96.9819724559784
pilot  :  2.988756448030472
engineer  :  0.029250176157802343
</pre>

<hr>
<br>

<img src="test-images/11.jpg" />
<pre>
engineer  :  100.0
pilot  :  8.049450689329163e-09
farmer  :  1.503418743664664e-09
</pre>

<hr>

<br>

<h3><b><u>References</u></b></h3>

 1. T. Gebru et al, Datasheets for Datasets, <br>
 <a href="https://arxiv.org/abs/1803.09010" >https://arxiv.org/abs/1803.09010</a> <br><br>
 2. Kaiming H. et al, Deep Residual Learning for Image Recognition <br>
 <a href="https://arxiv.org/abs/1512.03385" >https://arxiv.org/abs/1512.03385</a> <br><br>
