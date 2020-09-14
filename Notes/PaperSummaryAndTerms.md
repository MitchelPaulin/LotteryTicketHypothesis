# Paper Summary

## 1 Introduction

Pruned networks can reduce parameter-counts by 90%. This decreases size or energy consumption, without harming accuracy. Contemporarily, architectures uncovered from this pruning are harder to train from scratch and reach lower accuracy than the original.  
The paper aims to show **The Lottery Ticket Hypothesis**, which is within a randomly initialized dense neural network, there exists a subnet which, when trained in isolation, performs as well as the original with at most the same number of iterations (A Winning Ticket).  
Winning Tickets identified by training a neural network and pruning its smallest magnitude weights. Each un pruned connection's value is reset to initialization from before training.  
This pruning approach is *one-shot*. This paper focuses on *iterative pruning*, which trains, prunes, and resets the network over n rounds. each round prunes a percent of the survivors of the previous round. This finds better winning tickets than one-shot pruning.

*This "Iterative Pruning Method" is the basis for finding "Winning Tickets". Said tickets can then be trained in isolation of the original network in at most the same amount of time and provide at least the same amount of accuracy and efficiency.*

## 2 Winning Tickets in Fully-Connected Networks

Lottery Ticket Hypothesis is applied to fully-connected networks trained on MNIST. Pruning is done using a layer-wise heuristic, removing a percentage of weights with the lowest magnitudes at each layer.
___Iterative Pruning:___ Winning tickets learn faster than the original network. The performance of winning tickets is better than the original network until the network has been pruned to less than 3.6% of its original size.  
___Random Re initialization:___ When randomly reinitialize the winning tickets learn slower than the original network. This supports the hypothesis' emphasis on initialization, original initialization withstands pruning, while random reinitialization suffers.  
___One-Shot Pruning:___ Itereative pruning creates smaller winning tickets, they can be costly to find. One-shot pruning can identify winning tickets without the need for repeated training, however iteratively pruned tickets learn faster and are more accurate at smaller sizes. This aligns with the goal to identify the smallest winning ticket and is the focus of the paper.  

## 3 Winning Tickets in Convolutional Networks

Lottery Ticket Hypothesis is applied to convolutional networks, increasing complexity of the learning problem and size of the network. It is applied to Conv-2, Conv-4, and Conv-6 architectures, which have 2, 4, and 6 convolutional layers respectively, followed by two fully-connected layers.  
___Finding Winning Tickets:___ Winning tickets reach minimum validation loss between 2.5x - 3.5x faster (depending on the architecture). Test accuracy improves from 3.3% to 3.5%. All three networks remain above original test accuracy. The winning tickets also generalize better.
___Random Reinitialization:___ After a certain time the results are similar to those of the same test in section 2. However, test accuracy at early stopping are steady and improve for Conv-2 and Conv-4, meaning that at moderate pruning structure alone can lead to better accuracy
___Dropout:___ Dropout interacts with the Lottery Ticket Hypothesis in a complementary way, which could make winning tickets easier to find.  

## 4 VGG and RESNET for CIFAR10

Study of the Lottery Ticket Hypothesis in architectures used in practice. Winning tickets are still found, however iterative pruning is sensitive to learning rate. Early stopping time not measured, though accuracy is plotted at several points during training instead.  
___Global Pruning:___ For VGG and RESNET the lowest magnitude weights collectively across all layers is removed, since there is larger disparity in the amount of parameters on each layer. This identifies smaller winning tickets and avoids bottlenecks.  
___VGG:___ At higher learning rates, iterative pruning does not find winning tickets, however at lower learning rates the previous pattern holds. Bridging the gap is increasing the learning rate from 0 to the initial over k iterations.  
___RESNET:___ Results are similar to ___VGG___, however even with warmup, a winning ticket with the original learning rate of 0.1 could not be found.  

## 5 Discussion

___Winning Ticket Initialization:___ When randomly reinitialized, a winning ticket learns more slowly and gets lower test accuracy, suggesting that initialization is important for success. Networks pruned by 80% can be randomly reinitialized and trained to similar accuracy, however this does not hold for further pruning.  
___Winning Ticket Structure:___ Other research suggests that the structure of a network can determine the types of data it can separate more efficiently.  
___Improved Generalization for Winning Tickets:___ The best generalization comes from a model which has been pruned significantly but not too much, as models which can still be compressed offer the greatest generalization.  

## 6 Limitations

Extremely large datasets are not tested as iterative pruning is computationally intensive.  
Sparse pruning is the only method of finding winning tickets and the resulting architectures are not optimized for modern libraries or hardware.  
On deeper networks (RESNEt VGG) iterative pruning is unable to find winning tickets without learning rate warmup.  

## B Iterative Pruning Strategies

**Both strategies have detailed outlines in full paper**  
Strategy 1: Iterative Pruning with Resetting  
Strategy 2: Iterative Pruning with Continued Training  
The difference is that strategy 2 uses the already trained weights after every round of pruning whereas strategy 1 resets the weights. Weights are reset after sufficient pruning in both strategies. Strategy 1 maintains higher accuracy and faster early stopping times to smaller network sizes.

## F.5 Initial and Final Weights in Winning Tickets

Generally winning ticket weights are more likely to increase in magnitude than are weights that do not participate in the eventual winning ticket.

## G Hyperparameters for Fully-Connected Networks

This section explains the hyperparameters selected and evaluates how far lottery ticket experiment patterns extend to other hyperparameter choices.  

## G.3.1 SGD

SGD optimized networks can be pruned to smaller sizes with similar results as Adam optimized networks.  

## G.4 Iterative Pruning Rate

The paper uses a pruning rate of 0.2, with and when the output layer is small, use a 0.1 pruning rate for the output layer (as in all Lenet architecture tests).  

## G.5 Initialization Distribution

Paper uses only a Gaussian Glorot initialization scheme.  

## H Hyperparameters for Convolutional Networks

This section explains the hyperparameters selected and evaluates how far lottery ticket experiment patterns extend to other hyperparameter choices.  

## H.4 Iterative Pruning Rate

Different pruning rates required for convolutional and fully connected layers, seeking to not create any bottlenecks by pruning too much of a layer.  

## I Hyperparameters for VGG and RESNET on CIFAR10

This section explains the hyperparameters selected.  

## Terms

Pruning - Techniques for removing unnecessary weights from neural networks  

MNIST - Dataset of handwritten digits  

CIFAR10 - Common dataset for training convolutional neural networks, contains images of 10 different classes  

Feed-Forward Neural Network - Connections between nodes do not form a cycle, first and simplest type of neural network  

Stochastic Gradient Descent (SGD) - A learning rate optimization algorithm  

Minimum Validation Loss/Early Stopping Time - The point at which the network stops improving on some arbitrary hold out validation dataset  

Generalization - How well a neural network handles data it hasn't seen before  

Dropout - Improve accuracy by randomly disabling a fraction of the units (sampling a sub network) on each training iteration  

Adam - A learning rate optimization algorithm
