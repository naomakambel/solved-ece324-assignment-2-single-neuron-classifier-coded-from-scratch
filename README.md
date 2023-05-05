Download Link: https://assignmentchef.com/product/solved-ece324-assignment-2-single-neuron-classifier-coded-from-scratch
<br>
The goal of this assignment is to build a single-neuron classifier that solves the following problem: given a 3×3 array of binary data (only 1’s and 0’s), determine when the 3×3 pattern is an ‘X’ as illustrated in Figure 1. To be clear, the goal is to make a classifier that outputs a ‘1’ when the input is the pattern in Figure 1, and a ‘0’ when it is any other pattern of 1’s and 0’s in the 3×3 grid.

Figure 1: Problem Definition – ‘Recognize’ this Pattern

As noted, this is a simple problem to solve with the usual ‘procedural’ coding that you learned in first year – a simple if statement in Python. It is also possible to determine a correct answer for the linear classifier by inspection, as also discussed. Instead, the goal in this assignment is to gain an understanding of the ‘learning from data’ approach, and to use the specific method of learning employed in the successful deep learning approach: an artificial neural network trained with gradient descent. This will underpin your understanding when we apply various versions of this approach to much more difficult problems in later assignments and your course project.

<h1>1         Neural Network Classifier Method to Solve Problem</h1>

The neural net machine learning method is illustrated in Figure 2. It shows the nine inputs of the 3×3 array (the <em>I<sub>i </sub></em>in the figure) all being fed into a single artificial neuron, which computes the linear function <em>Z </em>= (<sup>P8</sup><em><sub>i</sub></em><sub>=0 </sub><em>w<sub>i</sub>I<sub>i</sub></em>) + <em>b</em>, and then passes it through an <em>activation function</em>, such as a sigmoid, ReLU or just linear (<em>Y </em>= <em>Z</em>) as described in class. The weights (<em>w<sub>i</sub></em>) and bias (<em>b</em>) must be determined, through a training process, so that the output <em>Y </em>correctly indicates whether the input pattern is the one shown in Figure 1 or not.

Figure 2: Single Neuron Classifier

The sections below will ask you to write the code implements this neural network computation an <em>trains </em>it to set the values of the weights (<em>w<sub>i</sub></em>) and bias (<em>b</em>) parameters.

<h1>2         Solve by Inspection</h1>

In class we discussed this structure above, using a ReLU activation function, and determined by inspection, values for the weights (<em>w<sub>i</sub></em>) and bias (<em>b</em>) parameters that would make the classifier work as described. For this to work, you must also say how to interpret the output, Z, of the classifier i.e. say in words what output values of Z indicate a ‘match’ with the required pattern and which indicate ‘no match.’ Answer the following questions relating to the problem of solving for the weights by inspection:

<ol>

 <li>Is the answer that we discussed in class unique? If your answer is yes, say why. If not, give a second answer that uses different weights and bias.</li>

 <li>How many unique inputs (that is, different instances of <em>I </em>= {<em>I</em><sub>0</sub><em>,I</em><sub>1</sub><em>,…,I</em><sub>8</sub>}) are possible for the 3×3 grid?</li>

</ol>

<ul>

 <li>Does your solution easily scale to solve a 4×4 problem, and an NxN problem? Explain why or why not.</li>

</ul>

<ol>

 <li>Suppose that, on a 5×5 grid, you had to match the ‘X’ as above, but, in addition, an ‘X’ shifted left by one, and shifted right by 1 position also had to be matched. Could you as easily create the single-neuron parameters to solve that problem? Why or why not?</li>

</ol>

<h1>3         Training from Data</h1>

You are to write a Python-based program, in a file called a2.py, using PyCharm, that trains the single-neuron classifier shown in Figure 2 using the gradient descent method as described in class. This includes the calculation of parameter (i.e. weight and bias) gradients through analytic equations. Your program should have input parameters that allow you to easily be able to select the following different choices or parameters (see below for a pointer on how to do this parameterization easily):

<ol>

 <li>The activation functions should be selectable as: sigmoid, ReLU and linear. (linear means <em>Y </em>= <em>Z </em>in Figure 2). Note that in class we derived the equations for the gradient of the weights and bias for the <em>linear </em>activation function, making use of the chain rule. While the derivative of the linear activation function is the constant one, the derivative of the sigmoid and ReLU functions is not as simple.</li>

 <li>The learning rate (as described in class).</li>

 <li>Number of epochs (number of times the entire training set is used to determine a modification to the weights and bias).</li>

 <li>Random number Seed (setting this value differently changes the random initialization of the weights).</li>

</ol>

As described in class you should use the separate files of data to train and then validate the training. These are provided for you in the associated assignment files named as follows:

<table width="327">

 <tbody>

  <tr>

   <td width="123">File</td>

   <td width="204">Contents</td>

  </tr>

  <tr>

   <td width="123">traindata.csv</td>

   <td width="204">200 example 3×3 grids</td>

  </tr>

  <tr>

   <td width="123">trainlabel.csv</td>

   <td width="204">labels for training examples</td>

  </tr>

  <tr>

   <td width="123">validdata.csv</td>

   <td width="204">20 different 3×3 grids</td>

  </tr>

  <tr>

   <td width="123">validlabel.csv</td>

   <td width="204">labels for validation examples</td>

  </tr>

 </tbody>

</table>

The data in the files is formatted as follows: the traindata.csv and validdata.csv files contain one example input per line, given as nine numbers, separated by commas, consecutively representing the inputs <em>I</em><sub>0</sub><em>,I</em><sub>1</sub><em>,…I</em><sub>8</sub>. The trainlabel.csv and validlabel.csv files contain one number per line, either 0 or 1, indicating if the corresponding line in the Data file is a match for the X pattern (with a value of 1) or not a match (with a value of 0). Be sure to view the files with a text editor of some kind to make sure you agree that the labels are correct.

You can use the numpy function loadtxt to read this data into your code.

<h1>4         Experiments and Outputs to Hand In</h1>

As you write and debug your code, you’ll need to test individual parts of your code in the usual way (by inspecting the output from subsections of the code, after setting the input). With machine learning, you will also need to inspect the <em>learning curves </em>which shown the progress of the classifier’s <em>loss </em>function and <em>accuracy </em>after each step. We are interested to see if the <em>training </em>loss and accuracy are improving after each epoch, and also whether the <em>validation </em>loss and accuracy are improving. So, in your code, use matplotlib to shown how loss and accuracy are changing vs. epoch. Be sure to put the training and validation loss on one plot (as is typical practice in the field), and the training and validation accuracy on a second plot.

You will need to experiment with the learning rate parameter to find one that works well, and to determine how many epochs are needed to succeed. (We expect to succeed for this problem by the way, since we know that there are good solutions as determined in Section 3.)

In your report, assign2.pdf you should hand in data, tabular form, that shows the effect of the following on the training and validation accuracy. For each item below, select (and report) reasonable values of the <em>other </em>parameters. (e.g. when showing the effect of epoch, choose fixed values for learning rate and activation function that give a good sense of what the variation in number of epochs does).

<ol>

 <li>The number of epochs.</li>

 <li>The learning rate – be sure to show a range where the learning rate is too high, and where it is too low.</li>

 <li>The effect of the three activation functions – linear, sigmoid and ReLU. Explain why the best one came out best.</li>

 <li>The effect of 5 different random seeds; for this choose a learning rate that isn’t the best, but one that does not succeed as well as your best. Explain why the answers differ.</li>

</ol>

Finally, you should determine a single set of parameters that achieve the best result you can get, but in the fewest epochs. Indicate in your report, what those parameters are, and what test and validation accuracy you achieved.

In addition, in you should create and hand in <em>properly labelled </em>loss and accuracy plots vs. epoch, as well as the nine weights displayed using the dispKernel function, as described above, for the following cases:

<ol>

 <li>Show an example where the learning rate is too <em>low </em>(which means learning is too slow). Give an explanation as to what is happening in this case.</li>

 <li>That shows an example where the learning rate is too <em>high</em>. Explain.</li>

 <li>That shows a ‘good’ learning rate.</li>

 <li>Give the two plots for the three activation functions, linear, sigmoid and ReLU.</li>

</ol>