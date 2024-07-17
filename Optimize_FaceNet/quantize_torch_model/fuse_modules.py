import torch
import torch.nn as nn
import copy
from facenet_pytorch import InceptionResnetV1

"""
Fusing layers combines them into a single operation, prevents excessive loss of accuracy during quantization and 
improves inference speed, fusible layers include:
    - conv (convolution) + bn (batch normalization)
    - conv +  bn + relu (activation function)
    - conv + relu
    - linear (fully connected layer) + bn
    - linear + relu
    
InceptionResnetV1 model architecture, 4 types of nested layers


Model
    Layer[0]
        Module [Conv, Bn, Relu]
    
    Layer[1]
        Sub-Layer[0]
            Module [Conv, Bn, Relu]
        .
        .
    Layer[2]
        Sub-Layer[0]
            Branch[0]
                Module [Conv, Bn, Relu]
            .
            .
        .
        .
    Layer[3]
        Sub-Layer[0]
            Branch[0]
                Sub-Branch[0]
                    Module [Conv, Bn, Relu]
                .
                .
            .
            .
        .
        .
    .
    .

"""
class Fusion:
    def __init__(self):
        self.fused_modules = [] #contains all the layers NAMES that have been fused
        self.modules = [] #contains all the modules in layer.sub-layer.branch.sub-branch format, ie model[7][0][1][0], so fused layers can be easily accessed again /
   
    def fuse(self):
        #blocks -> layers -> modules/branches 
        model = InceptionResnetV1(pretrained='vggface2').eval()
        fused_modules = []
        
        for i, (layer_name, layer) in enumerate(list(model.named_children())): #iterating over the most high level layers
            if self.is_empty(layer) == False:
                #print(list(layer.named_children())[0][0])
                has_sub_layers = isinstance(layer, nn.Sequential) or isinstance(list(layer.children())[1], nn.Sequential)
              
                if has_sub_layers:
                    for j, (sub_layer_name, sub_layer) in enumerate(layer.named_children()):
                        has_branches = True #after examining architecture, sub_layers always have several branches
                        
                        if has_branches:
                            for l, (branch_name, branch) in enumerate(sub_layer.named_children()):
                                has_sub_branches = isinstance(branch, nn.Sequential) 
                                
                                if has_sub_branches:
                                    for t, (sub_branch_name, sub_branch) in enumerate(branch.named_children()):
                                        
                                        if self.is_empty(sub_branch) == False:
                                            fused_modules = self.fuse_this_layer(sub_branch)
                                            self.write_fused_modules(layer_name, sub_layer_name, branch_name, sub_branch_name, fused_modules=fused_modules)
                                            self.modules.append([i, j, l, t])
                                else:
                                    if self.is_empty(branch) == False:
                                        fused_modules = self.fuse_this_layer(branch)
                                        self.write_fused_modules(layer_name, sub_layer_name, branch_name, fused_modules=fused_modules)
                                        self.modules.append([i, j, l])
                        else:
                            fused_modules= self.fuse_this_layer(sub_layer)
                            self.write_fused_modules(layer_name, sub_layer_name, fused_modules=fused_modules)
                            self.modules.append([i, j])
                else:
                    fused_modules = self.fuse_this_layer(layer)
                    self.write_fused_modules(layer_name, fused_modules=fused_modules)
                    self.modules.append([i])
        """         
        print('Fused modules:')
        [print(fused_module) for fused_module in self.fused_modules]
        """
        return model
    
    def fuse_this_layer(self, layer): #no nested layers, this layer contains (conv, bn, relu)
        modules_to_fuse = []
    
        for name, module in list(layer.named_children()): #get names of modules in this layer, (ie conv, bn, relu)
            modules_to_fuse.append(name)    
        torch.quantization.fuse_modules(layer, modules_to_fuse, inplace=True) #fuse modules

        return modules_to_fuse

    def write_fused_modules(self, *args, fused_modules): #adds the fused_modules and which layer (nested path included) they were from
        layer_name = args[0]
        for layer in args[1:]:
            layer_name += '.' + str(layer)
            
        self.fused_modules.append(f'{layer_name}: Fused {fused_modules}')
        
    def is_empty(self, layer): #check if the layer is empty, or does not contain fusable modules
        if layer == ():
            print('empty')
            return True
        return len(list(layer.children())) < 3 
        
if __name__ == '__main__':
    model = InceptionResnetV1(pretrained='vggface2').eval()
    #print(model)
    fusion = Fusion()
    model = fusion.fuse()
    model.eval()
    