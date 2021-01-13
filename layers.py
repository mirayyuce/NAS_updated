import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import utils
import copy


class Concatenate(torch.nn.Module):
    """
    Merge by concatenation class
    """
    def __init__(self):
        super(Concatenate, self).__init__()
        
    def forward(self, input1, input2):
        """
        Forward propagation for concatenation. 
        
        Args:
            input1: layer 1
            input2: layer 2

        Returns:
            Returns concatenated output 
        """
        return torch.cat((input1, input2), dim=1)


class ConcatenateConvex(torch.nn.Module):
    """
    Merge by convex combination class
    """
    def __init__(self):
        super(ConcatenateConvex, self).__init__()
        # Learnable parameter lambda
        self._lambda = nn.Parameter(torch.zeros(1,1).cuda())
        
    def forward(self, input1, input2):
        """
        Forward propagation for convex combination. 
        
        Args:
            input1: layer 1
            input2: layer 2

        Returns:
            Returns convex combinated output 
        """
        # We add weight to the concatenated connection
        output = (torch.ones(1,1).cuda() - self._lambda) * input1 + self._lambda * input2
        return output



class AddLayer(torch.nn.Module):
    """
    Split up block class. It does not add a new layer. 
    """
    def __init__(self):
        super(AddLayer, self).__init__()

    def forward(self, input1, input2):
        """
        Forward propagation for split up connection. 
        
        Args:
            input1: layer 1
            input2: layer 2

        Returns:
            Returns sum of two layers
        """
        output = input1 + input2
        return output

class depthwise_separable_conv(torch.nn.Module):
    """
    Separable convolution class
    """
    def __init__(self, nin, nout, kernel_size, padding):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        """
        Forward propagation for sep conv
        
        Args:
            x: input to layer

        Returns:
            Returns sep conv output
        """
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvNet(torch.nn.Module):
    """
    The main class creating the model using model_description. 
    Basic net structure with train and eval and so on
    Custom methods shoudl use this
    """
    def __init__(self, model_descriptor):
        """
        Initializator of the ConvNet class
        
        Args:
            model_descriptor: model descriptor

        Returns:
        """
        super(ConvNet, self).__init__()

        self.model_descriptor = copy.deepcopy(model_descriptor)
        self.topo_ordering = []  
        self.alpha = model_descriptor['compile']['optimizer']['alpha']

        self.layerdic = {}

        self.init_layers()

        # Pptimizer parameters
        m = self.model_descriptor['compile']['optimizer']['momentum']
        lr = self.model_descriptor['compile']['optimizer']['lr']
        self.optimizer = self.model_descriptor['compile']['optimizer']['name'](self.parameters(), lr=lr, momentum=m)

        # Scheduler params 
        T_max = self.model_descriptor['compile']['scheduler']['T_max']
        eta_min = self.model_descriptor['compile']['scheduler']['eta_min']
        last_epoch = self.model_descriptor['compile']['scheduler']['last_epoch']
        self.scheduler = self.model_descriptor['compile']['scheduler']['name'](self.optimizer, T_max=T_max,
                                                                              eta_min=eta_min, last_epoch=last_epoch)

        self.loss = self.model_descriptor['compile']['loss']()

    def init_layers(self):
        """
        Function to initialize layers and build the topological ordering 
        Args:
        Returns:
        """
        model_descriptor_layers_copy = copy.deepcopy(self.model_descriptor['layers'])
        
        # Start building the network with input layers
        layers2process = [layer for layer in model_descriptor_layers_copy if layer['type'] == 'input']

        while len(layers2process) > 0:

            layer_added = True

            # Get next layer to process
            layer = layers2process.pop(0)
            layer['id'] = str(layer['id'])
            # Take first input
            layer['input'][0] = str(layer['input'][0])

            # Don't do anything for the input layer, we just needed the id. It doesn't have any functionality
            # it's just a starting point for the chain
            if layer['type'] == 'input':
                pass

            elif layer['type'] == 'conv':
                # We emulate padding "same" behavior
                padding1 = (int(layer['params']['ks1']) - 1) // 2
                padding2 = (int(layer['params']['ks2']) - 1) // 2

                # Use the base class' add_module method. It adds a layer to the network without connecting. 
                # Topological order connects them for forward and backward propagate
                self.add_module(str(layer['id']), nn.Conv2d(layer['params']['in_channels'],
                                                            layer['params']['channels'],
                                                            kernel_size=(
                                                                layer['params']['ks1'],
                                                                layer['params']['ks2']),
                                                            padding=(padding1, padding2)))
            elif layer['type'] == 'sep':
                padding1 = (int(layer['params']['ks1']) - 1) // 2
                padding2 = (int(layer['params']['ks2']) - 1) // 2
               
                self.add_module(str(layer['id']), depthwise_separable_conv(layer['params']['in_channels'],
                                                                layer['params']['channels'],
                                                                kernel_size=(
                                                                    layer['params']['ks1'],
                                                                    layer['params']['ks2']),
                                                                padding=(padding1, padding2)))

            elif layer['type'] == 'batchnorm':
                self.add_module(str(layer['id']), nn.BatchNorm2d(num_features=layer['params']['in_channels']))

            elif layer['type'] == 'activation':
                self.add_module(str(layer['id']), nn.ReLU())

            elif layer['type'] == 'pool':

                if layer['params']['pooltype'] == 'avg':
                    self.conv.add_module(str(layer['id']), nn.AvgPool2d(kernel_size=layer['params']['poolsize']))

                elif layer['params']['pooltype'] == 'max':
                    self.add_module(str(layer['id']), nn.MaxPool2d(kernel_size=layer['params']['poolsize']))

            elif layer['type'] == 'dense':
                # In this work we have only one dense layer. 
                # Dense layer has a fixed id agp, not a int id
                
                # We can't use a linear layer just after the conv layer. We need to reshape it. First we average it

                # Before applying the dense layer, we need to average input using average pooling
                # This layer is fixed and has the constant name
                self.add_module("agp", nn.AvgPool2d(layer['params']['in_size']))

                self.add_module(str(layer['id']),
                                nn.Linear(layer['params']['in_channels'], layer['params']['units']))

                self.output_id = layer['id']
                self.last_channels = layer['params']['in_channels']

            elif layer['type'] == 'merge':

                # For all merge layers there are two input layers required
                # They must be already in the added modules before we add the "merge layer"
                if (str(layer['input'][0]) in self.topo_ordering) and (
                        str(layer['input'][1]) in self.topo_ordering) and (str(layer['id']) not in self.topo_ordering):

                    if layer['params']['mergetype'] == 'concat':
                        self.add_module(str(layer['id']), Concatenate())

                    elif layer['params']['mergetype'] == 'convcomb':
                        self.add_module(str(layer['id']), ConcatenateConvex())

                    elif layer['params']['mergetype'] == 'add':
                        self.add_module(str(layer['id']), AddLayer())

                else:
                    # Wait until both input layers are added (we can check topo_ordering)
                    layer_added = False

            if layer_added:
                # Append all layers whose input node = current layer
                # We consider to add some layers, but we need to make sure that all parent layers should be in the
                # Topological ordering already, otherwise wait
                layers2process.extend(
                    [subsequent_layers for subsequent_layers in copy.deepcopy(self.model_descriptor['layers']) if
                     int(layer['id']) in subsequent_layers['input']])

                
                self.topo_ordering.append(str(layer['id']))


    def forward(self, x):
        """
        Forward method of the custom network. We connect the layers here 
        
        Args:
            x: input to layer

        Returns:
            Returns model output
        """
        model_descriptor_layers_copy = copy.deepcopy(self.model_descriptor['layers'])

        # Start with the input layers
        layers2process = [inputlayer for inputlayer in model_descriptor_layers_copy if inputlayer['type'] == 'input']

        # This is our main dictionary which allows to receive the output from any layer
        self.layerdic = {}

        while len(layers2process) > 0:

            # Get next layer to process
            layer = layers2process.pop(0)
            layer['id'] = str(layer['id'])
            layer['input'][0] = str(layer['input'][0])

            layer_added = True

            if layer['type'] == 'input':
                self.layerdic[layer['id']] = x

            elif layer['type'] == 'dense':
                # We have one avg pool operation, it's not a layer
                agp_result = self._modules["agp"](self.layerdic[layer['input'][0]])
                # Flat layer 
                flat_result = agp_result.view(-1, self.last_channels)
                # We need to flatten it because it's multi dimensional
                dense_result = self._modules[str(layer['id'])](flat_result)
                self.layerdic[layer['id']] = dense_result

            elif layer['type'] == 'merge':
                # For all merge layers there are two input layers required
                # They must be already at layerdic before we can receive the output from the merge layer
                if (str(layer['input'][0]) in self.layerdic.keys()) and (str(layer['input'][1]) in self.layerdic.keys()) \
                        and (str(layer['id']) not in self.layerdic.keys()):
                    # These layers has two inputs
                    self.layerdic[layer['id']] = self._modules[str(layer['id'])](
                        self.layerdic[str(layer['input'][0])], self.layerdic[str(layer['input'][1])])

                else:
                    # Wait until both input layers are added (we can check topo_ordering)
                    layer_added = False

            else:  
                # For the other type of layers just call forward function (=call())
                # self._modules[]() calls forward (witout a dot, calls __call__)
                # self._modules[].forward(parent_output)
                self.layerdic[layer['id']] = self._modules[str(layer['id'])](self.layerdic[layer['input'][0]])

            if layer_added:
                # Append all layers whose input node = current layer
                layers2process.extend(
                    [subsequent_layers for subsequent_layers in copy.deepcopy(self.model_descriptor['layers']) if
                     int(layer['id']) in subsequent_layers['input']])

                # Last layer so far
                self.output_id = layer['id']

        output = self.layerdic[self.output_id]
        return output

    def train(self, x_val, y_val):
        """
        Trains the network and backpropagates
        
        Args:
            x_val: input to layer as minibatches
            y_val: labels

        Returns:
            Returns model output and total loss
        """
        '''
        fit -> train -> forward -> returns output -> calculate loss-> train returns loss
        '''
        x = Variable(x_val, requires_grad=False)
        y = Variable(y_val, requires_grad=False)

        # Mixup prepare
        x, y_a, y_b, lam = utils.mixup_data(x, y, self.alpha)

        x = Variable(x, requires_grad=False)
        y_a = Variable(y_a, requires_grad=False)
        y_b = Variable(y_b, requires_grad=False)
        
        self.optimizer.zero_grad()

        output = self.forward(x)

        # Mixup criterion - supervisor wanted this loss
        loss_mixup = lam * self.loss(output, y_a) + (1 - lam) * self.loss(output, y_b)

        # Weight decay 
        L2_decay_sum = 0        
        for name, param in self.named_parameters():
            if 'weight' in name:  
                name_id = str(name.split('.')[0]) 
                # Get the string name of the layer - layer type
                layer_name = copy.deepcopy(self._modules[name_id].__class__.__name__)
                if layer_name == 'Conv2d' or layer_name == 'Linear' or layer_name == "depthwise_separable_conv":
                    L2_decay_sum += 0.0005 * torch.norm(param.view(-1),2) # Regularization

        # Total loss
        loss_loc = loss_mixup + L2_decay_sum
        
        # Updates the parameters at the end of the minibatch

        loss_loc.backward(retain_graph=True) 

        # Update the optimizer
        self.optimizer.step() 

        return output, loss_loc.data

    def fit(self, trainloader, train_type, epochs=1, scheduler=optim.lr_scheduler.CosineAnnealingLR):
        """
        Fits the input to the winner model. Calls the train function. Updates the scheduler.
        
        Args:
            trainloader: trainloader object of pytorch
            train_type: flag to differentiate training of vanilla model, child network and the winner network
            epochs: number of epochs to train
            scheduler: learning rate schedular

        Returns:
            Returns model output and total loss
        """
        
        loss_arr = []

        if train_type == 'winner':
            # Initialize a new scheduler for the final training of the winner
            n_minibatches = len(trainloader)
            sch_epochs = epochs * n_minibatches - 1
            # Set the scheduler parameters
            self.scheduler = scheduler(self.optimizer, T_max=sch_epochs, eta_min=0, last_epoch=-1)

        for i in range(epochs):
            loss_arr = []
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                
                # Take a scheduler step if the model is not vanilla
                if train_type != 'vanilla':
                    self.scheduler.step()
                
                data, target = Variable(inputs.cuda()), Variable(targets.cuda())

                outputs, loss = self.train(data, target)

                loss_arr.append(loss)

                if batch_idx % 50 == 0:
                    print(batch_idx, loss)

    def predict(self, x_val):
        """
        Function to predict the model output
        
        Args:
            x_val: input to the model
        Returns: 
            Predicted output
        """
        outputs = self.forward(x_val)
        # Return to cpu for the next steps
        temp_cpu_output = outputs.cpu()
        return temp_cpu_output.data.numpy().argmax(axis=1)

    def evaluate(self, testloader):
        """
        Function to evaluate the model performance
        
        Args:
            testloader: Pytorch testloader object
        Returns: 
            Accuracy percentage
        """
        acc = 0
        set_size = 0
        for inputs_test, targets_test in testloader:
            data_test, target_test = Variable(inputs_test.cuda()), Variable(targets_test.cuda())

            y_hat = self.predict(data_test)
            y_hat_np = np.array(y_hat)
            target_test_cpu = target_test.cpu()
            target_test_np = target_test_cpu.numpy()

            acc += np.sum(y_hat_np == target_test_np)
            set_size += len(inputs_test)

        acc = 100 * acc / set_size
        return acc
