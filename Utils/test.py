import glob
import os
images_path = "./Datasets/CUB_200_2011/images/"
segs_path = "./Datasets/CUB_200_2011/segmentations/"


images = glob.glob( 
    os.path.join(images_path,"**/*.jpg")) + \
    glob.glob( os.path.join(images_path,"**/*.png")  ) +  \
    glob.glob( os.path.join(images_path,"**/*.jpeg")  )
segmentations  =  glob.glob( os.path.join(segs_path,"**/*.png")  ) 

base_name = [os.path.basename(i) for i in segmentations]
ret = []

for im in images:
    seg_bnme = os.path.basename(im).replace(".jpg" , ".png").replace(".jpeg" , ".png")
    # seg = os.path.join( segs_path , seg_bnme  )
    assert ( seg_bnme in base_name ),  (im + " is present in "+images_path +" but "+seg_bnme+" is not found in "+segs_path + " . Make sure annotation image are in .png"  )
    ret.append((im , segmentations[base_name.index(seg_bnme)]) )
    break
