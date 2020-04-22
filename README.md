# GRNUlar
Gene Regulatory Network reconstruction using Unrolled algorithms  (Paper link: TBD)

Explore the notebook folder for a quick overview of the GRNUlar algorithm ran on jupyter notebook.  

The 'grnular' folder is a python module of the algorithm implementation.  

Please do not hesitate to connect with me via email or raise an issue on Github in case of any query.  

# GRNUlar Architecture based on Unrolled algorithm concept

![architectureNN](https://github.com/Harshs27/GRNUlar/blob/master/architecture_images/grnular_architecture1.png)  
Fig.1 Using the neural network in a multi-task learning framework which act asnon-linear regression functions between TFs and other genes. We start with a fullyconnected NN indicating all genes are dependent on all the input TFs (dotted blacklines).  Assume that in the process of discovering the underlying sparse GRN ouralgorithm zeroes out all the edge weights except the blue ones. Now, if there is apath from an input TF to an output gene, then we conclude that the output gene isdependent on the corresponding input TF.  

![architectureUnrolled](https://github.com/Harshs27/GRNUlar/blob/master/architecture_images/grnular_architecture2.png)   
Fig.2 Visualizing the GRNUlar algorithm’s unrolled architecture.  
