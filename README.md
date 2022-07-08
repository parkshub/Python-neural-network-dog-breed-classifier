# Convolutional Neural Network: Dog Breed Classifier


![Cover photo](/img/readme_dog_photo.jpg?raw=true "Cover Photo")
*photo by @marliesebrandsma*

## Background
For this project, I set out on a personal endeavor to create a dog breed classifier (133 breeds) with an accuracy at or above 90% using Python. Throughout this project, I explored various models and model structures, beginning with ResNet-18 and ending with a cocktail of concatenated convolutional neural networks (CCCNN). The initial exploratory stages—where I attempt to build my own ResNet18 and ResNet50—were imperative for my personal growth, although, less interesting. So I’ve condensed first three sections to quickly jump to section 4, where I explore the use of bottleneck features and compare the results to a regular model. Please, stick around until my favorite section, “Jessie and Friends.

## Discussion
Overall, it was a difficult, but rewarding project. Getting to look at pictures of dogs was also a great stress-reliever. If I weren't constrained by my laptop's hardware, there are several things I would have done that would have drastically improved the accuracy of my model. For one, I would have combined larger networks. I could have also used the newly released Efficient Nets. Unfortunately, they weren't available when embarking on this project. I could have also used larger input images. During resizing, some images were affected more than others and a significant amount of noise was added to the pictures.

![comparison photo](/img/comparison.png?raw=true "comparison photo")

Lastly, another idea I had was to integrate a model like You Only Look Once (YOLO) to draw square anchor boxes around the dogs and extracting only the image inside the box. By limiting the anchor boxes to a square, we can resize the images into a standard size without skewing the proportions of a dog's features, which is probably pretty important for discerning a dog's breed. I would have loved to have pursued this idea if time permitted. Nonetheless, it's in my list of projects to complete. Be on the lookout!

Please visit https://neural-network-breed-classifier.netlify.app/ for the full report.
