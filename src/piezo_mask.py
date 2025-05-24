import torch

def space_group_mask(material):
    mask = torch.tensor([[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]])
    if int(material['space_group_number']) in set(range(111,123)): #tet 4b2m
        mask = torch.tensor([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
    elif int(material['space_group_number']) in set(range(81,83)): #tet-4b
        mask = torch.tensor([[0,0,0,1,1,0],[0,0,0,1,1,0],[1,1,0,0,0,1]])
    elif int(material['space_group_number']) in set(range(94,111)): #hex-6mm tet-4mm
        mask = torch.tensor([[0,0,0,0,1,0],[0,0,0,1,0,0],[1,1,1,0,0,0]])
    elif int(material['space_group_number']) in set(range(75,81)): #hex-6 tet-4
        mask = torch.tensor([[0,0,0,1,1,0],[0,0,0,1,1,0],[1,1,1,0,0,0]])
    elif int(material['space_group_number']) == 174: #hex-6b
        mask = torch.tensor([[1,1,0,0,0,1],[1,1,0,0,0,1],[0,0,0,0,0,0]])
    elif int(material['space_group_number']) in set(range(187,191)): #hex-6bm2
        mask = torch.tensor([[0,0,0,0,0,1],[1,1,0,0,0,0],[0,0,0,0,0,0]])
    elif int(material['space_group_number']) in set(range(183,187)): #hex-6mm tet-4mm
        mask = torch.tensor([[0,0,0,0,1,0],[0,0,0,1,0,0],[1,1,1,0,0,0]])
    elif int(material['space_group_number']) in set(range(168,174)): #hex-6 tet-4
        mask = torch.tensor([[0,0,0,1,1,0],[0,0,0,1,1,0],[1,1,1,0,0,0]])
    elif int(material['space_group_number']) in set(range(195,200)): #cub-4b3m-23
        mask = torch.tensor([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
    elif int(material['space_group_number']) in set(range(215,221)): #cub-4b3m-23
        mask = torch.tensor([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
    elif int(material['space_group_number']) in set(range(156,161)): #tri-3m
        mask = torch.tensor([[0,0,0,0,1,1],[1,1,0,1,0,0],[1,1,1,0,0,0]])
    elif int(material['space_group_number']) in set(range(149,156)): #tri-32
        mask = torch.tensor([[1,1,0,1,0,0],[0,0,0,0,1,1],[0,0,0,0,0,0]])
    elif int(material['space_group_number']) in set(range(143,147)): #tri-3
        mask = torch.tensor([[1,1,0,1,1,1],[1,1,0,1,1,1],[1,1,1,0,0,0]])
    elif int(material['space_group_number']) in set(range(25,47)): #ort-mm2
        mask = torch.tensor([[0,0,0,0,1,0],[0,0,0,1,0,0],[1,1,1,0,0,0]])
    elif int(material['space_group_number']) in set(range(16,25)): #ort-222
        mask = torch.tensor([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
    elif int(material['space_group_number']) in set(range(6,10)): #mon-m
        mask = torch.tensor([[1,1,1,0,1,0],[0,0,0,1,0,1],[1,1,1,0,1,0]])
    elif int(material['space_group_number']) in set(range(3,6)): #mon-2
        mask = torch.tensor([[0,0,0,1,0,1],[1,1,1,0,1,0],[0,0,0,1,0,1]])
    elif int(material['space_group_number']) == 1: #tri-1
        mask = torch.tensor([[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]])
    else:
        print(f'space group {material["space_group_number"]} not recognized as piezoelectric')
    return mask
