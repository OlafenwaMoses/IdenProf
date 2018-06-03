# IdenProf
IdenProf is a dataset containing images of identifiable professionals.
<hr>
<b>IdenProf</b> is a dataset of identifiable professionals, collected in order to ensure that machine learning systems can be trained
 to recognize uniformed professionals as humans can observe. This is part of our mission to train machine learning systems to 
  perceive, understand and act accordingly in any environment they are deployed. <br><br>

  This is the first release of the IdenProf dataset. It contains 11,000 images that span cover 10 categories of professions. The professions 
  included in this release are: <br><br>

  - <b> Chef </b> <br>
  - <b> Doctor </b> <br>
  - <b> Engineer </b> <br>
  - <b> Farmer </b> <br>
  - <b> Fireman </b> <br>
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
 You can download the train and test images via this <a href="https://github.com/OlafenwaMoses/IdenProf/releases/" >link</a> . <br><br>

 We have also provided a python codebase to download the images, train <b>ResNet50</b> on the images
  and perform prediction using a pretrained model (also using <b>ResNet50</b>) provided in the release section of this repository.
  The python codebase is contained in the <b><a href="idenprof.py" >idenprof.py</a></b> file and the model class labels for prediction is also provided the 
  <b><a href="idenprof_model_class.json" >idenprof_model_class.json</a></b>. The pretrained <b>ResNet50</b> model is available for download via this 
  <b><a href="https://github.com/OlafenwaMoses/IdenProf/releases/" >link</a></b>.
<br><br> <br> <br>

  <b>>>> DATASHEET FOR IDENPROF</b> <br><br>
  For transparency and accountability on the collection and content of the IdenProf dataset, we have provided a comprehensive
   datasheet on the dataset . The datasheet is based on the blueprint provided in the 2018 paper publication , "Datasheets for Datasets" by Timnit. et al.
    The datasheet is available via this <b><a href="idenprof-datasheet.pdf" >link</a></b>.

<br>

<h3><b><u>References</u></b></h3>

 1. T. Gebru et al, Datasheets for Datasets, <br>
 <a href="https://arxiv.org/abs/1803.09010" >https://arxiv.org/abs/1803.09010</a> <br><br>
 2. Kaiming H. et al, Deep Residual Learning for Image Recognition <br>
 <a href="https://arxiv.org/abs/1512.03385" >https://arxiv.org/abs/1512.03385</a> <br><br>
