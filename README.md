# Covid-19-classification-by-pytorch

In this repository, I reviewed the respotiry created by Shervin Minaee, Milan Sonka (the previous editor in chief of IEEE TMI), Rahele Kafieh, Shakib Yazdani, and Ghazaleh Jamalipour Soufi,  to make their work to a practical app. The paper can be seen <a href= "https://arxiv.org/pdf/2004.09363.pdf"> here </a>.

Besides, I separate their work to three sections and add more details to explain the procedure. This code clarified the procedures to be more clear and comprehensible.


Besides, I adjust the code in a way that can be used in notebook which is more interactive, where you can run code, visualize data and include text all in one document.

# Upload the whole dataset in jupyter notebook

We can upload the dataset including train dataset and test dataset:<a href= "https://github.com/ieee8023/covid-chestxray-dataset"> Covid-chestxray-dataset </a> and <a href= "https://stanfordmlgroup.github.io/competitions/chexpert"> ChexPert Dataset </a>  in one single zip file in jupytor noteook, then extract it as folder in your notebook. By this way, you can upload whole dataset in your notebook. 


# Train the model using pretrained model Resnet 18

I trained the model for 30 epoch and saved the model, but the jupiter code only shows the result of 3 epoches. 

# Test the model.
I test the model for 100 covid images and 100 non-covid images, the results showed that the confusion matrix workes well to disniguish non-covid cases, but covid cases results is pretty wrong disnguished. I am going to find the root causes.
