

def conv(file_input): 
    import torchvision 
    import torchvision.transforms as transforms
    import os
    from PIL import Image
    import cv2
    import torch
    directory = 'files\\'
    path_img = 0
    with os.scandir(path=directory) as it:
       
            
                print("dir: \t" + file_input)
                path_img = directory + file_input
                img_obj = Image.open(path_img)
                img_obj = img_obj.resize((32, 32))
                img_obj.save(path_img)


    image_use = cv2.imread(path_img)
    image_use = cv2.cvtColor(image_use, cv2.COLOR_BGR2RGB)
    users_transform = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    img_tensor = users_transform(image_use)
    img_float_tensor = torch.unsqueeze(img_tensor, 0)
    #print("tensor for input:", img_float_tensor)
    os.remove('files//test.jpg')
    return img_float_tensor

#print(conv("car.jpg"))